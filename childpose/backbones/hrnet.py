# Adapted from HRNet-Human-Pose-Estimation (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch,
# MIT License, Copyright (c) Microsoft, written by Bin Xiao), ``lib/models/pose_hrnet.py``.
# W48 COCO body without the COCO heatmap layer, inference only.
"""HRNet-W48 encoder: the fused 1/4-resolution, 48-channel map pooled to a token grid and projected to ``dim``."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

# (num_modules, num_branches, block, num_blocks, num_channels) of the W48 stages 2-4.
STAGES = {
    2: (1, 2, "BASIC", (4, 4), (48, 96)),
    3: (4, 3, "BASIC", (4, 4, 4), (48, 96, 192)),
    4: (3, 4, "BASIC", (4, 4, 4, 4), (48, 96, 192, 384)),
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


BLOCKS = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):
    def __init__(
        self, num_branches: int, block, num_blocks: Sequence[int], num_inchannels: List[int],
        num_channels: Sequence[int], multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = nn.ModuleList(
            [self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)]
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, i: int, block, num_blocks, num_channels) -> nn.Sequential:
        downsample = None
        if self.num_inchannels[i] != num_channels[i] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[i], num_channels[i] * block.expansion, 1, bias=False),
                nn.BatchNorm2d(num_channels[i] * block.expansion),
            )
        layers = [block(self.num_inchannels[i], num_channels[i], 1, downsample)]
        self.num_inchannels[i] = num_channels[i] * block.expansion
        layers += [block(self.num_inchannels[i], num_channels[i]) for _ in range(1, num_blocks[i])]
        return nn.Sequential(*layers)

    def _make_fuse_layers(self) -> nn.ModuleList:
        nb, ch = self.num_branches, self.num_inchannels
        fuse_layers = []
        for i in range(nb if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(nb):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(ch[j], ch[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(ch[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        out_ch = ch[i] if last else ch[j]
                        mods = [nn.Conv2d(ch[j], out_ch, 3, 2, 1, bias=False), nn.BatchNorm2d(out_ch)]
                        if not last:
                            mods.append(nn.ReLU(True))
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
                y = y + (x[j] if i == j else self.fuse_layers[i][j](x[j]))
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNetW48(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        pre = [256]
        for s in (2, 3, 4):
            num_modules, num_branches, block, num_blocks, num_channels = STAGES[s]
            block = BLOCKS[block]
            channels = [c * block.expansion for c in num_channels]
            setattr(self, f"transition{s - 1}", self._make_transition_layer(pre, channels))
            modules = []
            for m in range(num_modules):
                multi_scale = not (s == 4 and m == num_modules - 1)  # stage 4 ends with the high-res branch only
                modules.append(HighResolutionModule(num_branches, block, num_blocks, channels, num_channels, multi_scale))
                channels = modules[-1].num_inchannels
            setattr(self, f"stage{s}", nn.Sequential(*modules))
            pre = channels

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

    def _make_layer(self, block, planes: int, blocks: int) -> nn.Sequential:
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, 1, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

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


class HRNetBackbone(nn.Module):
    has_cls_token = False

    def __init__(self, dim: int, token_grid_hw: Tuple[int, int], feature_channels: int) -> None:
        super().__init__()
        self.hrnet = HRNetW48()
        self.hrnet_token_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(tuple(token_grid_hw)), nn.Conv2d(feature_channels, dim, kernel_size=1)
        )
        self.grid_hw = tuple(token_grid_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hrnet_token_adapter(self.hrnet(x)).flatten(2).transpose(1, 2)
