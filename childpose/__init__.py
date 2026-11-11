"""ChildPose: memory-augmented 2D pose estimation for children in video (inference code)."""

from .config import COCO_KEYPOINTS, COCO_SKELETON, VARIANTS, ChildPoseConfig
from .model import ChildPoseModel, load_model
from .predictor import ChildPosePredictor

__all__ = [
    "COCO_KEYPOINTS",
    "COCO_SKELETON",
    "VARIANTS",
    "ChildPoseConfig",
    "ChildPoseModel",
    "ChildPosePredictor",
    "load_model",
]
