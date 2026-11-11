"""Video and image inference for one tracked child at a time."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from .head import decode_heatmaps
from .memory import MemoryState
from .model import ChildPoseModel, load_model
from .preprocess import crop_and_normalize, crop_box, crop_to_image, to_pil

Box = Sequence[float]
Boxes = Union[Sequence[Optional[Box]], Mapping]
"""Per-frame child box ``[x1, y1, x2, y2]`` in pixels, ``None`` where the child is not visible.
A list indexed by frame, or a mapping ``{frame_index: box}``."""


class ChildPosePredictor:
    """Top-down 2D pose (COCO-17) of one child from frames and the child's per-frame box.

    Args:
        model: a :class:`ChildPoseModel`, the path of a released ``.safetensors`` checkpoint, or a variant
            name such as ``"vitpose_base"`` (downloaded from the Hugging Face Hub).
        device: defaults to CUDA when available.
        amp: float16 autocast on CUDA (default ``True`` on CUDA).
        batch_size: frames encoded together by the backbone.
    """

    def __init__(
        self,
        model: Union[ChildPoseModel, str, Path],
        device: Optional[Union[str, torch.device]] = None,
        amp: Optional[bool] = None,
        batch_size: int = 8,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(model, ChildPoseModel):
            self.model = model.to(self.device).eval()
        else:
            self.model = load_model(model, self.device)
        self.amp = (self.device.type == "cuda") if amp is None else bool(amp and self.device.type == "cuda")
        self.batch_size = int(batch_size)
        self._out_wh = self.model.crop_wh

    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _prepare(self, frame: Any, box: Box):
        cfg = self.model.cfg
        image = to_pil(frame)
        cb = crop_box(box, image.width, image.height, cfg.bbox_expand)
        return crop_and_normalize(image, cb, self._out_wh, cfg.mean, cfg.std), cb

    def _decode(self, heatmaps: torch.Tensor, boxes) -> List[Dict[str, Any]]:
        xy, scores = decode_heatmaps(heatmaps.float(), self._out_wh)
        out = []
        for k, cb in enumerate(boxes):
            kp = crop_to_image(xy[k], cb, self._out_wh).cpu().numpy()
            sc = scores[k].cpu().numpy()
            out.append({"keypoints": kp, "scores": sc, "score": float(sc.mean()), "crop_box": list(cb)})
        return out

    @torch.no_grad()
    def predict_video(
        self,
        frames: Iterable[Any],
        boxes: Boxes,
        fps: Optional[float] = None,
        mode: str = "interleave",
        memory_fps: float = 5.0,
        use_memory: bool = True,
        clip_len: Optional[int] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Pose for the frames of a video of any length, in one pass in frame order.

        ``frames`` is any iterable of PIL images or RGB ``uint8`` arrays (a generator works, so long videos
        are never held in memory). Returns one entry per input frame: ``None`` for frames without a box or
        not predicted (see ``mode``), else a dict with ``keypoints`` ``[17, 2]`` (image pixels), per-joint
        ``scores`` ``[17]`` (heatmap peaks), ``score`` (their mean) and the ``crop_box`` used.

        The models were trained on frames sampled at about ``memory_fps`` (5 fps): each frame attends to up
        to 6 earlier frames about 0.2 s apart. With the input frame rate ``fps`` and
        ``step = round(fps / memory_fps)``, ``mode`` sets how a faster video is fed to the memory:

        - ``"interleave"`` (default): every frame is predicted; frame ``i`` reads and writes memory
          ``i % step``, so its memory holds frames ``i - step``, ``i - 2*step``, ... (about 0.2 s, 0.4 s, ...
          earlier), as in training. Each frame is still encoded once.
        - ``"downsample"``: only frames ``0, step, 2*step, ...`` are predicted (the video at about 5 fps, as
          in training); the other frames get ``None``.
        - ``"sequential"``: every frame is predicted with a single memory of the previous frames at the
          input frame rate (frames ``i - 1``, ``i - 2``, ...), i.e. closer in time than in training.

        With ``fps=None`` the input is taken to be at about 5 fps already (``step = 1``) and the three modes
        are the same. The memory runs over the whole video; ``clip_len`` empties each memory every
        ``clip_len`` frames it has seen. Frames without a box are skipped and do not enter the memory.
        ``use_memory=False`` predicts every frame (every kept frame for ``"downsample"``) on its own.
        """
        if mode not in ("interleave", "downsample", "sequential"):
            raise ValueError(f"mode must be 'interleave', 'downsample' or 'sequential', got {mode!r}")
        step = 1 if fps is None else max(1, int(round(float(fps) / float(memory_fps))))
        keep_every = step if mode == "downsample" else 1
        streams = step if mode == "interleave" else 1
        fifos: List[MemoryState] = [[] for _ in range(streams)]
        seen = [0] * streams
        results: List[Optional[Dict[str, Any]]] = []
        pending: List[tuple] = []

        def flush() -> None:
            if not pending:
                return
            idx, crops, cboxes = zip(*pending)
            pending.clear()
            with self._autocast():
                tokens = self.model.encode(torch.stack(crops).to(self.device))
                if use_memory:
                    conditioned = []
                    for k, i in enumerate(idx):
                        s = i % streams
                        if clip_len and seen[s] % int(clip_len) == 0:
                            fifos[s] = []
                        z, fifos[s] = self.model.step(tokens[k: k + 1], fifos[s])
                        seen[s] += 1
                        conditioned.append(z)
                    tokens = torch.cat(conditioned, dim=0)
                heatmaps = self.model.heatmaps(tokens)
            for i, res in zip(idx, self._decode(heatmaps, cboxes)):
                results[i] = res

        for i, frame in enumerate(frames):
            results.append(None)
            if i % keep_every:
                continue
            box = boxes.get(i) if isinstance(boxes, Mapping) else (boxes[i] if i < len(boxes) else None)
            if box is None:
                continue
            crop, cb = self._prepare(frame, box)
            pending.append((i, crop, cb))
            if len(pending) >= self.batch_size:
                flush()
        flush()
        return results

    @torch.no_grad()
    def predict_images(self, images: Sequence[Any], boxes: Sequence[Box]) -> List[Dict[str, Any]]:
        """Single-image pose (no temporal memory), one box per image."""
        results: List[Dict[str, Any]] = []
        for start in range(0, len(images), self.batch_size):
            items = [self._prepare(im, bx) for im, bx in zip(images[start: start + self.batch_size], boxes[start: start + self.batch_size])]
            crops, cboxes = zip(*items)
            with self._autocast():
                heatmaps = self.model.heatmaps(self.model.encode(torch.stack(crops).to(self.device)))
            results.extend(self._decode(heatmaps, cboxes))
        return results


__all__ = ["ChildPosePredictor"]
