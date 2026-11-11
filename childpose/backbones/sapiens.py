# ViT encoder in the layout of MMPretrain ``VisionTransformer`` (https://github.com/open-mmlab/mmpretrain,
# Apache-2.0) as used by Meta Sapiens (https://github.com/facebookresearch/sapiens, CC BY-NC 4.0).
"""Sapiens ViT encoder (Meta Sapiens 0.3B / 0.6B / 1B), inference only.

Parameter names follow the original Sapiens (mmpretrain ``VisionTransformer``) checkpoints.
Output: ``[N, 1 + Hp*Wp, D]`` tokens, CLS first, patch tokens in row-major order.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# (embed_dim, depth, num_heads); patch size 16 and MLP ratio 4 for every size.
SAPIENS_ARCHS = {
    "sapiens_0.3b": (1024, 24, 16),
    "sapiens_0.6b": (1280, 32, 16),
    "sapiens_1b": (1536, 40, 24),
}
PATCH_SIZE = 16


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        # ``layers.0.0`` / ``layers.1`` match the parameter names of the Sapiens checkpoints.
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden)), nn.Linear(hidden, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[1](F.gelu(self.layers[0](x)))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FFN(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class SapiensViT(nn.Module):
    def __init__(self, arch: str, input_hw: Tuple[int, int]) -> None:
        super().__init__()
        embed_dim, depth, num_heads = SAPIENS_ARCHS[arch]
        self.embed_dim = embed_dim
        self.grid_hw = (input_hw[0] // PATCH_SIZE, input_hw[1] // PATCH_SIZE)
        self.patch_embed = PatchEmbed(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid_hw[0] * self.grid_hw[1], embed_dim))
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads) for _ in range(depth)])
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.ln1(x)


class SapiensBackbone(nn.Module):
    has_cls_token = True

    def __init__(self, arch: str, input_hw: Tuple[int, int]) -> None:
        super().__init__()
        self.encoder = SapiensViT(arch, input_hw)
        self.grid_hw = self.encoder.grid_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
