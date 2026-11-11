"""Heatmap pose head: spatial tokens -> conv stem -> two x2 deconvolutions -> one heatmap per joint."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class PoseHead(nn.Module):
    def __init__(self, dim: int, num_joints: int, grid_hw: Tuple[int, int], has_cls_token: bool) -> None:
        super().__init__()
        self.grid_hw = tuple(grid_hw)
        self.has_cls_token = bool(has_cls_token)
        c1 = max(dim // 2, 256)
        c2 = max(c1 // 2, 128)
        self.stem = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False), _group_norm(dim), nn.GELU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(dim, c1, 4, stride=2, padding=1, bias=False), _group_norm(c1), nn.GELU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(c1, c2, 4, stride=2, padding=1, bias=False), _group_norm(c2), nn.GELU())
        self.head = nn.Conv2d(c2, num_joints, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` ``[N, num_tokens, dim]`` -> heatmaps ``[N, J, 4*Hf, 4*Wf]``. The CLS token is not used."""
        if self.has_cls_token:
            tokens = tokens[:, 1:]
        n, _, d = tokens.shape
        hf, wf = self.grid_hw
        x = tokens.reshape(n, hf, wf, d).permute(0, 3, 1, 2)
        return self.head(self.up2(self.up1(self.stem(x))))


def decode_heatmaps(heatmaps: torch.Tensor, crop_wh: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Arg-max decoding. Returns ``xy`` ``[..., J, 2]`` in crop pixels (cell centers) and ``scores`` ``[..., J]``
    (peak heatmap value)."""
    hh, wh = heatmaps.shape[-2:]
    flat = heatmaps.flatten(-2)
    scores, idx = flat.max(dim=-1)
    x = (idx % wh).to(heatmaps.dtype)
    y = (idx // wh).to(heatmaps.dtype)
    xs = (x + 0.5) / float(wh) * float(crop_wh[0])
    ys = (y + 0.5) / float(hh) * float(crop_wh[1])
    return torch.stack([xs, ys], dim=-1), scores
