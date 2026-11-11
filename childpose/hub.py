"""Download released checkpoints from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .config import VARIANTS

HF_REPO_ID = "Asp1rinn/ChildPose"
"""Hugging Face model repository that hosts the checkpoints."""


def checkpoint_filename(variant: str) -> str:
    return f"checkpoints/childpose_{variant}.safetensors"


def download_checkpoint(
    variant: str,
    repo_id: str = HF_REPO_ID,
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Download (or reuse from the local Hugging Face cache) the checkpoint of ``variant``; returns its path."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("downloading checkpoints needs `pip install huggingface_hub`") from e
    return Path(hf_hub_download(repo_id, checkpoint_filename(variant), revision=revision, cache_dir=cache_dir))


def resolve_checkpoint(name_or_path: Union[str, Path]) -> Path:
    """A local file path is returned as is; a variant name (e.g. ``"vitpose_base"``) is downloaded."""
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return path
    if str(name_or_path) in VARIANTS:
        return download_checkpoint(str(name_or_path))
    raise FileNotFoundError(f"{name_or_path} is neither a checkpoint file nor a variant name {sorted(VARIANTS)}")
