"""Checkpoint management for LipRead.

Saves and restores model weights, optimizer state, scheduler state, AMP scaler,
and training progression metadata to E:\\LipRead_Data\\checkpoints\\.
Checkpoints are never stored inside the Git repository.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.data.paths import get_checkpoint_dir

logger = logging.getLogger(__name__)


def get_default_checkpoint_dir() -> Path:
    """Returns the verified checkpoint directory on E: drive."""
    ckpt_dir = get_checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool = False,
    checkpoint_dir: Optional[Path] = None,
    filename: str = "latest_model.pt",
) -> Tuple[Path, Optional[Path]]:
    """Saves training checkpoint containing model and optimizer states.

    Args:
        state: Dictionary with model_state_dict, optimizer_state_dict, epoch, etc.
        is_best: If True, also saves a copy as best_model.pt.
        checkpoint_dir: Target directory. Defaults to E:\\LipRead_Data\\checkpoints.
        filename: Name for the latest checkpoint file (default: latest_model.pt).

    Returns:
        Tuple of (latest_checkpoint_path, best_checkpoint_path if is_best else None).
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_default_checkpoint_dir()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / filename
    torch.save(state, latest_path)
    logger.info(f"Saved latest checkpoint: {latest_path} (epoch {state.get('epoch', 0)})")

    best_path: Optional[Path] = None
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
        logger.info(f"Saved best checkpoint: {best_path} (best val loss: {state.get('best_validation_loss', float('inf')):.4f})")

    return latest_path, best_path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    """Loads a checkpoint from disk and restores weights into model and optimizer.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: PyTorch model instance to load weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional learning rate scheduler to restore state into.
        scaler: Optional PyTorch AMP GradScaler to restore state into.
        device: Target device to map tensors to.

    Returns:
        Dictionary containing checkpoint metadata (epoch, global_step, best_validation_loss, config, etc.).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    map_location = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: Dict[str, Any] = torch.load(path, map_location=map_location)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if raw state dict was saved
        model.load_state_dict(checkpoint)
    logger.info(f"Loaded model weights from {path}")

    # Restore optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Restored optimizer state.")

    # Restore scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored learning rate scheduler state.")

    # Restore AMP scaler if provided
    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("Restored GradScaler state.")

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_validation_loss": checkpoint.get("best_validation_loss", float("inf")),
        "config": checkpoint.get("config", {}),
        "random_seed": checkpoint.get("random_seed", None),
        "metrics": checkpoint.get("metrics", {}),
    }

    return metadata
