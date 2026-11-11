"""Top-down cropping: enlarge the child's box, crop, resize to the model input, normalize; and the inverse map."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

BoxXYXY = Tuple[float, float, float, float]


def to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """PIL image, or an RGB ``uint8`` array ``[H, W, 3]``."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")


def _clip(box: Sequence[float], width: int, height: int) -> BoxXYXY:
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
    y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def crop_box(box_xyxy: Sequence[float], width: int, height: int, expand: float) -> BoxXYXY:
    """Clip the box to the image, enlarge it by ``expand`` about its center, clip again."""
    x1, y1, x2, y2 = _clip(box_xyxy, width, height)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w, h = (x2 - x1) * expand, (y2 - y1) * expand
    return _clip((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), width, height)


def crop_and_normalize(
    image: Image.Image, box: BoxXYXY, out_wh: Tuple[int, int], mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """Crop ``box`` (rounded to pixels), bicubic-resize to ``out_wh`` and normalize -> ``[3, H, W]`` float32."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, image.width)), max(0, min(x2, image.width))
    y1, y2 = max(0, min(y1, image.height)), max(0, min(y2, image.height))
    if x2 <= x1 or y2 <= y1:
        crop = Image.new("RGB", out_wh, color=(128, 128, 128))
    else:
        crop = image.crop((x1, y1, x2, y2)).resize(out_wh, resample=Image.BICUBIC)
    x = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    return (x - mean_t) / std_t


def crop_to_image(xy_crop: torch.Tensor, box: BoxXYXY, crop_wh: Tuple[int, int]) -> torch.Tensor:
    """Map ``[..., 2]`` points from crop pixels back to image pixels."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    x = xy_crop[..., 0] / float(crop_wh[0]) * bw + x1
    y = xy_crop[..., 1] / float(crop_wh[1]) * bh + y1
    return torch.stack([x, y], dim=-1)
