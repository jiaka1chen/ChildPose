# Adapted from HRFormer (https://github.com/HRNet/HRFormer, MIT License, Copyright (c) 2021 Microsoft,
# written by Rao Fu, RainbowSecret, Lang Huang), ``pose/mmpose/models/backbones/hrt.py`` and its modules.
# The interlaced window attention derives from openseg.pytorch (https://github.com/openseg-group/openseg.pytorch, MIT).
# HRT-Base COCO 384x288 body, inference only, without mmcv / mmpose.
"""HRFormer-Base (HRT) encoder: the 1/4-resolution, 78-channel map pooled to a token grid and projected to ``dim``."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# HRT-Base COCO 384x288: (num_modules, num_branches, num_blocks, num_channels, num_heads, mlp_ratios)
STAGES = {
    2: (1, 2, (2, 2), (78, 156), (2, 4), (4, 4)),
    3: (4, 3, (2, 2, 2), (78, 156, 312), (2, 4, 8), (4, 4, 4)),
    4: (2, 4, (2, 2, 2, 2), (78, 156, 312, 624), (2, 4, 8, 16), (4, 4, 4, 4)),
}
WINDOW_SIZE = 7


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, downsample: nn.Module = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class WindowAttention(nn.Module):
    """Multi-head self-attention over the tokens of each local window (``[L, B', C]``, sequence first)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length, bsz, dim = x.shape
        head_dim = dim // self.num_heads
        q = self.q_proj(x) * float(head_dim) ** -0.5
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.contiguous().view(length, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        attn = torch.bmm(q, k.transpose(1, 2)).softmax(dim=-1)
        out = torch.bmm(attn, v).transpose(0, 1).contiguous().view(length, bsz, dim)
        return self.out_proj(out)


class InterlacedPoolAttention(nn.Module):
    """Local-window self-attention: center-pad to a multiple of the window, group windows, attend, undo."""

    def __init__(self, dim: int, num_heads: int, window_size: int = WINDOW_SIZE) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = WindowAttention(dim, num_heads)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        ws = self.window_size
        pad_h = math.ceil(h / ws) * ws - h
        pad_w = math.ceil(w / ws) * ws - w
        x = x.view(b, h, w, c)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        hp, wp = h + pad_h, w + pad_w
        qh, qw = hp // ws, wp // ws
        x = x.view(b, qh, ws, qw, ws, c).permute(2, 4, 0, 1, 3, 5).reshape(ws * ws, b * qh * qw, c)
        out = self.attn(x)
        out = out.view(ws, ws, b, qh, qw, c).permute(2, 3, 0, 4, 1, 5).reshape(b, hp, wp, c)
        if pad_h > 0 or pad_w > 0:
            out = out[:, pad_h // 2: pad_h // 2 + h, pad_w // 2: pad_w // 2 + w, :]
        return out.reshape(b, n, c)


class MlpDWBN(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(hidden)
        self.dw3x3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.norm2 = nn.BatchNorm2d(hidden)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.permute(0, 2, 1).contiguous().reshape(b, c, h, w)
        x = self.act(self.norm1(self.fc1(x)))
        x = self.act(self.norm2(self.dw3x3(x)))
        x = self.act(self.norm3(self.fc2(x)))
        return x.reshape(b, c, -1).permute(0, 2, 1).contiguous()


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.attn = InterlacedPoolAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MlpDWBN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1).contiguous()
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x.permute(0, 2, 1).contiguous().view(b, c, h, w)


class HighResolutionTransformerModule(nn.Module):
    def __init__(
        self, num_blocks: Sequence[int], channels: Sequence[int], num_heads: Sequence[int],
        mlp_ratios: Sequence[float], multiscale_output: bool,
    ) -> None:
        super().__init__()
        self.num_branches = len(channels)
        self.branches = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(channels[i], num_heads[i], mlp_ratios[i]) for _ in range(num_blocks[i])])
            for i in range(self.num_branches)
        ])
        self.fuse_layers = self._make_fuse_layers(channels, multiscale_output)
        self.relu = nn.ReLU(inplace=True)

    def _make_fuse_layers(self, ch: Sequence[int], multiscale_output: bool) -> nn.ModuleList:
        nb = self.num_branches
        fuse_layers = []
        for i in range(nb if multiscale_output else 1):
            fuse_layer = []
            for j in range(nb):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(ch[j], ch[i], kernel_size=1, bias=False),
                        nn.BatchNorm2d(ch[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="bilinear", align_corners=False),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        out_ch = ch[i] if last else ch[j]
                        mods = [
                            nn.Conv2d(ch[j], ch[j], kernel_size=3, stride=2, padding=1, groups=ch[j], bias=False),
                            nn.BatchNorm2d(ch[j]),
                            nn.Conv2d(ch[j], out_ch, kernel_size=1, bias=False),
                            nn.BatchNorm2d(out_ch),
                        ]
                        if not last:
                            mods.append(nn.ReLU(inplace=True))
                        convs.append(nn.Sequential(*mods))
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]), size=tuple(x[i].shape[2:]), mode="bilinear", align_corners=False
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        downsample = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256))
        self.layer1 = nn.Sequential(Bottleneck(64, 64, downsample), Bottleneck(256, 64))

        pre = [256]
        for s in (2, 3, 4):
            num_modules, _nb, num_blocks, channels, num_heads, mlp_ratios = STAGES[s]
            setattr(self, f"transition{s - 1}", self._make_transition_layer(pre, list(channels)))
            setattr(self, f"stage{s}", nn.Sequential(*[
                HighResolutionTransformerModule(
                    num_blocks, channels, num_heads, mlp_ratios,
                    multiscale_output=not (s == 4 and m == num_modules - 1),  # stage 4 ends with the high-res branch only
                )
                for m in range(num_modules)
            ]))
            pre = list(channels)

    @staticmethod
    def _make_transition_layer(pre: List[int], cur: List[int]) -> nn.ModuleList:
        layers = []
        for i in range(len(cur)):
            if i < len(pre):
                if cur[i] != pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre[i], cur[i], 3, 1, 1, bias=False), nn.BatchNorm2d(cur[i]), nn.ReLU(inplace=True)
                    ))
                else:
                    layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - len(pre)):
                    out_ch = cur[i] if j == i - len(pre) else pre[-1]
                    convs.append(nn.Sequential(
                        nn.Conv2d(pre[-1], out_ch, 3, 2, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
                    ))
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y_list = [self.layer1(x)]
        for s, transition in ((2, self.transition1), (3, self.transition2), (4, self.transition3)):
            x_list = []
            for i in range(STAGES[s][1]):
                if transition[i] is not None:
                    x_list.append(transition[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = getattr(self, f"stage{s}")(x_list)
        return y_list[0]


class HRFormerBackbone(nn.Module):
    has_cls_token = False

    def __init__(self, dim: int, token_grid_hw: Tuple[int, int], feature_channels: int) -> None:
        super().__init__()
        self.hrt = HRT()
        self.hrt_token_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(tuple(token_grid_hw)), nn.Conv2d(feature_channels, dim, kernel_size=1)
        )
        self.grid_hw = tuple(token_grid_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hrt_token_adapter(self.hrt(x)).flatten(2).transpose(1, 2)
