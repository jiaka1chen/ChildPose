"""Model configurations of the released ChildPose variants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ChildPoseConfig:
    variant: str
    backbone: str
    """One of ``sapiens_0.3b``, ``sapiens_0.6b``, ``sapiens_1b``, ``vitpose_base``, ``hrnet_w48``, ``hrformer_base``."""
    input_hw: Tuple[int, int]
    """(height, width) of the person crop fed to the backbone."""
    dim: int
    """Token width (backbone output, memory and pose head)."""
    token_grid_hw: Optional[Tuple[int, int]] = None
    """HRNet / HRFormer only: the high-resolution feature map is pooled to this token grid."""
    feature_channels: Optional[int] = None
    """HRNet / HRFormer only: channels of the high-resolution branch."""
    mem_dim: int = 512
    max_memory_frames: int = 6
    memory_num_layers: int = 1
    memory_num_heads: int = 16
    num_joints: int = 17
    bbox_expand: float = 1.1
    """The person box is enlarged by this factor about its center before cropping."""
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "ChildPoseConfig":
        raw: Dict[str, Any] = json.loads(text)
        known = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
        return cls(**kw)


VARIANTS: Dict[str, ChildPoseConfig] = {
    c.variant: c
    for c in [
        ChildPoseConfig("hrnet_w48", "hrnet_w48", (288, 384), 1024, (16, 16), 48),
        ChildPoseConfig("hrformer_base", "hrformer_base", (288, 384), 1024, (16, 16), 78),
        ChildPoseConfig("vitpose_base", "vitpose_base", (256, 192), 768),
        ChildPoseConfig("sapiens_0.3b", "sapiens_0.3b", (1024, 1024), 1024),
        ChildPoseConfig("sapiens_0.6b", "sapiens_0.6b", (1024, 1024), 1280),
        ChildPoseConfig("sapiens_1b", "sapiens_1b", (1024, 1024), 1536),
    ]
}

# COCO-17 keypoint order used by every model.
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)
