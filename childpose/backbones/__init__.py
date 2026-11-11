"""Image encoders. Each maps ``[N, 3, H, W]`` crops to ``[N, num_tokens, dim]`` tokens and exposes
``grid_hw`` (spatial token grid) and ``has_cls_token`` (a CLS token precedes the spatial tokens)."""

from __future__ import annotations

import torch.nn as nn

from ..config import ChildPoseConfig


def build_backbone(cfg: ChildPoseConfig) -> nn.Module:
    if cfg.backbone.startswith("sapiens_"):
        from .sapiens import SapiensBackbone

        return SapiensBackbone(cfg.backbone, cfg.input_hw)
    if cfg.backbone == "vitpose_base":
        from .vitpose import ViTPoseBackbone

        return ViTPoseBackbone(cfg.input_hw)
    if cfg.backbone == "hrnet_w48":
        from .hrnet import HRNetBackbone

        return HRNetBackbone(cfg.dim, cfg.token_grid_hw, cfg.feature_channels)
    if cfg.backbone == "hrformer_base":
        from .hrformer import HRFormerBackbone

        return HRFormerBackbone(cfg.dim, cfg.token_grid_hw, cfg.feature_channels)
    raise ValueError(f"unknown backbone {cfg.backbone!r}")


__all__ = ["build_backbone"]
