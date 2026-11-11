"""Temporal memory: each frame attends to the encoded tokens of up to ``max_memory_frames`` past frames.

The memory is a FIFO of per-frame memory slots. Step ``t`` reads only slots of frames before ``t``,
each tagged with a learned embedding of its lag (1 = previous frame). Nothing depends on a fixed
sequence length, so a video of any length is processed one frame at a time.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

MemoryState = List[torch.Tensor]
"""FIFO of memory slots ``[B, N, mem_dim]``, oldest first. Start a new track with an empty list."""


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MemoryAttentionBlock(nn.Module):
    """Self-attention over the current frame's tokens, cross-attention to the memory, FFN (all pre-norm)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        if memory is not None:
            h = self.norm2(x)
            x = x + self.cross_attn(h, memory, memory, need_weights=False)[0]
        return x + self.mlp(self.norm3(x))


class MemoryModule(nn.Module):
    def __init__(self, dim: int, mem_dim: int, max_memory_frames: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.max_memory_frames = int(max_memory_frames)
        # Memory encoder: frame tokens -> memory slot (``mem_encoder.net``: LN, Linear, GELU, Linear).
        self.mem_encoder = nn.Module()
        self.mem_encoder.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, max(mem_dim * 2, dim)), nn.GELU(), nn.Linear(max(mem_dim * 2, dim), mem_dim)
        )
        # Learned embedding of the lag k = 1..max_memory_frames (``time_emb.table``).
        self.time_emb = nn.Module()
        self.time_emb.table = nn.Embedding(self.max_memory_frames, mem_dim)
        self.mem_to_attn = nn.Linear(mem_dim, dim)
        self.norm_mem = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([MemoryAttentionBlock(dim, num_heads) for _ in range(num_layers)])

    def _memory_tokens(self, fifo: MemoryState, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not fifo:
            return None
        past = fifo[-self.max_memory_frames:]
        lag_emb = self.time_emb.table.weight.to(dtype)  # row k-1 = lag k
        parts = [slot + lag_emb[lag - 1].view(1, 1, -1) for lag, slot in enumerate(reversed(past), start=1)]
        return self.norm_mem(self.mem_to_attn(torch.cat(parts, dim=1)))

    def step(self, x_t: torch.Tensor, fifo: MemoryState) -> Tuple[torch.Tensor, MemoryState]:
        """One frame. ``x_t``: tokens ``[B, N, dim]``. Returns memory-conditioned tokens and the updated FIFO."""
        slot = self.mem_encoder.net(x_t)
        memory = self._memory_tokens(fifo, x_t.dtype)
        z = x_t
        for layer in self.layers:
            z = layer(z, memory)
        fifo = (list(fifo) + [slot])[-self.max_memory_frames:]
        return z, fifo
