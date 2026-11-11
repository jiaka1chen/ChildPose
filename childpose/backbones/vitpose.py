# Adapted from ViTPose (https://github.com/ViTAE-Transformer/ViTPose, Apache-2.0),
# ``mmpose/models/backbones/vit.py``: ViT-B encoder only, inference only.
"""ViTPose-B encoder. Output: ``[N, Hp*Wp, 768]`` patch tokens (no CLS), row-major."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

EMBED_DIM, DEPTH, NUM_HEADS, PATCH_SIZE = 768, 12, 12, 16


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, -1))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, padding=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), (int(x.shape[2]), int(x.shape[3]))


class ViTPoseViT(nn.Module):
    def __init__(self, input_hw: Tuple[int, int]) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        num_patches = (input_hw[0] // PATCH_SIZE) * (input_hw[1] // PATCH_SIZE)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, EMBED_DIM))
        self.blocks = nn.ModuleList([Block(EMBED_DIM, NUM_HEADS) for _ in range(DEPTH)])
        self.last_norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.patch_embed(x)
        # ViTPose adds the CLS position embedding to every patch and has no CLS token.
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        return self.last_norm(x)


class ViTPoseBackbone(nn.Module):
    has_cls_token = False

    def __init__(self, input_hw: Tuple[int, int]) -> None:
        super().__init__()
        self.vitpose = ViTPoseViT(input_hw)
        # The padded patch conv gives floor((H + 4 - 16) / 16) + 1 rows, i.e. H // 16 for these sizes.
        self.grid_hw = ((input_hw[0] + 4 - PATCH_SIZE) // PATCH_SIZE + 1, (input_hw[1] + 4 - PATCH_SIZE) // PATCH_SIZE + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vitpose(x)
