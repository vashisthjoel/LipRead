"""Path utilities and helpers for dataset management in LipRead.

Provides standard resolution for raw video files, preprocessed tensors,
metadata annotations, and checkpoints across the configured data root.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from src.utils.config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    METADATA_DIR,
    MLFLOW_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    ensure_directories,
)
from src.utils.logging_config import get_logger

logger = get_logger("lipread.paths")


def get_project_root() -> Path:
    """Returns the code repository root directory."""
    return PROJECT_ROOT


def get_data_root() -> Path:
    """Returns the primary dataset and artifact root directory."""
    return DATA_ROOT


def get_raw_data_dir() -> Path:
    """Returns the raw video and audio dataset directory."""
    return RAW_DATA_DIR


def get_processed_data_dir() -> Path:
    """Returns the preprocessed lip crops and tensor directory."""
    return PROCESSED_DATA_DIR


def get_metadata_dir() -> Path:
    """Returns the dataset metadata and split files directory."""
    return METADATA_DIR


def get_checkpoint_dir() -> Path:
    """Returns the model weights and checkpoint directory."""
    return CHECKPOINT_DIR


def get_mlflow_dir() -> Path:
    """Returns the MLflow tracking experiments directory."""
    return MLFLOW_DIR


# Alias for MLflow directory
get_mlruns_dir = get_mlflow_dir


def ensure_data_directories(verbose: bool = False) -> Dict[str, Path]:
    """Ensures all target data directories exist on the storage volume."""
    return ensure_directories(verbose=verbose)


def get_raw_video_path(
    relative_path: Union[str, Path],
    dataset_subdir: Optional[str] = None,
) -> Path:
    """Resolves the absolute path for a raw input video file.

    Args:
        relative_path: Subpath or filename relative to raw data directory.
        dataset_subdir: Optional dataset subdirectory (e.g., 'lrs2_v1').

    Returns:
        Absolute Path object.
    """
    base = RAW_DATA_DIR / dataset_subdir if dataset_subdir else RAW_DATA_DIR
    return (base / relative_path).resolve()


def get_processed_tensor_path(
    relative_path: Union[str, Path],
    extension: str = ".pt",
) -> Path:
    """Resolves the absolute path for a processed tensor file.

    Args:
        relative_path: Relative path or identifier.
        extension: Expected tensor file extension (default: '.pt').

    Returns:
        Absolute Path object.
    """
    path = PROCESSED_DATA_DIR / relative_path
    if not path.suffix and extension:
        path = path.with_suffix(extension)
    return path.resolve()


def get_metadata_file_path(filename: str) -> Path:
    """Resolves the path to a metadata or annotation file.

    Args:
        filename: Name of the metadata file (e.g. 'train.txt', 'vocab.json').

    Returns:
        Absolute Path object.
    """
    return (METADATA_DIR / filename).resolve()


def get_checkpoint_path(experiment_name: str, filename: str) -> Path:
    """Resolves the path for a specific model checkpoint.

    Args:
        experiment_name: Name or ID of the training run.
        filename: Checkpoint filename (e.g. 'best_model.pt').

    Returns:
        Absolute Path object.
    """
    exp_dir = CHECKPOINT_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return (exp_dir / filename).resolve()


def check_storage_status() -> Dict[str, Any]:
    """Inspects the storage volume hosting DATA_ROOT.

    Returns:
        Dictionary reporting accessibility, total space, used space, and free space.
    """
    target = DATA_ROOT
    try:
        # Check volume drive
        drive = target.drive or str(target.resolve())[:3]
        total, used, free = shutil.disk_usage(str(target) if target.exists() else drive)
        return {
            "path": str(target),
            "accessible": True,
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
        }
    except Exception as exc:
        logger.warning("Could not read storage metrics for %s: %s", target, exc)
        return {
            "path": str(target),
            "accessible": target.exists(),
            "error": str(exc),
        }
