"""ChildPose model: image encoder -> token LayerNorm -> temporal memory -> heatmap pose head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

from .backbones import build_backbone
from .config import ChildPoseConfig
from .head import PoseHead
from .memory import MemoryModule, MemoryState

CHECKPOINT_FORMAT = "childpose-release-v1"


def checkpoint_metadata(cfg: ChildPoseConfig) -> dict:
    """safetensors metadata: a single key, so the file header (and its sha256) is deterministic."""
    info = {"format": CHECKPOINT_FORMAT, "variant": cfg.variant, "config": json.loads(cfg.to_json())}
    return {"childpose": json.dumps(info, sort_keys=True)}


class ChildPoseModel(nn.Module):
    def __init__(self, cfg: ChildPoseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.backbone_out_ln = nn.LayerNorm(cfg.dim)
        self.memory_module = MemoryModule(
            cfg.dim, cfg.mem_dim, cfg.max_memory_frames, cfg.memory_num_layers, cfg.memory_num_heads
        )
        self.pose_head = PoseHead(cfg.dim, cfg.num_joints, self.backbone.grid_hw, self.backbone.has_cls_token)

    @property
    def crop_wh(self) -> Tuple[int, int]:
        return self.cfg.input_hw[1], self.cfg.input_hw[0]

    def encode(self, crops: torch.Tensor) -> torch.Tensor:
        """Normalized crops ``[N, 3, H, W]`` -> frame tokens ``[N, num_tokens, dim]``."""
        return self.backbone_out_ln(self.backbone(crops))

    def step(self, tokens: torch.Tensor, fifo: MemoryState) -> Tuple[torch.Tensor, MemoryState]:
        """Condition one frame's tokens ``[B, num_tokens, dim]`` on the memory and return the updated memory."""
        return self.memory_module.step(tokens, fifo)

    def heatmaps(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tokens ``[N, num_tokens, dim]`` -> heatmaps ``[N, J, Hh, Wh]``."""
        return self.pose_head(tokens)

    def forward(self, clips: torch.Tensor, use_memory: bool = True) -> torch.Tensor:
        """Clips ``[B, T, 3, H, W]`` (memory starts empty at frame 0) -> heatmaps ``[B, T, J, Hh, Wh]``.

        With ``use_memory=False`` every frame is predicted on its own and the memory module is skipped.
        """
        b, t = clips.shape[:2]
        tokens = self.encode(clips.flatten(0, 1))
        if use_memory:
            tokens = tokens.view(b, t, *tokens.shape[1:])
            fifo: MemoryState = []
            out = []
            for i in range(t):
                z, fifo = self.step(tokens[:, i], fifo)
                out.append(z)
            tokens = torch.stack(out, dim=1).flatten(0, 1)
        hm = self.heatmaps(tokens)
        return hm.view(b, t, *hm.shape[1:])


def load_model(checkpoint: Union[str, Path], device: Union[str, torch.device] = "cpu") -> ChildPoseModel:
    """Build a model from a released ``.safetensors`` checkpoint, given as a file path or a variant name
    (``"vitpose_base"``, ``"sapiens_1b"``, ...; downloaded from the Hugging Face Hub). The model
    configuration is stored in the checkpoint."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    from .hub import resolve_checkpoint

    path = resolve_checkpoint(checkpoint)
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    info = json.loads(meta.get("childpose", "{}"))
    if info.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a ChildPose release checkpoint")
    model = ChildPoseModel(ChildPoseConfig.from_json(json.dumps(info["config"])))
    missing, unexpected = model.load_state_dict(load_file(str(path)), strict=False)
    # BatchNorm batch counters are not stored (they only matter for training).
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(device).eval()


__all__ = ["ChildPoseModel", "load_model", "CHECKPOINT_FORMAT"]
