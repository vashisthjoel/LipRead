"""Optimizer and learning rate scheduler utilities for LipRead.

Provides factory functions for AdamW optimizer and ReduceLROnPlateau scheduler.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer_type: str = "AdamW",
) -> Optimizer:
    """Builds and returns the optimizer for the model parameters.

    Args:
        model: PyTorch model whose parameters are to be optimized.
        lr: Learning rate (default: 1e-4).
        weight_decay: Weight decay / L2 regularization (default: 1e-4).
        optimizer_type: Name of the optimizer ('AdamW' or 'Adam').

    Returns:
        Configured PyTorch Optimizer.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'AdamW' or 'Adam'.")

    logger.info(
        f"Initialized {optimizer.__class__.__name__} (lr={lr}, weight_decay={weight_decay}, "
        f"parameters={len(trainable_params)})"
    )
    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "plateau",
    mode: str = "min",
    factor: float = 0.5,
    patience: int = 2,
    min_lr: float = 1e-6,
    verbose: bool = True,
) -> Optional[Any]:
    """Builds and returns the learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_type: 'plateau' (ReduceLROnPlateau) or 'none'.
        mode: Optimization mode ('min' for loss, 'max' for score).
        factor: Multiplicative factor of learning rate reduction.
        patience: Number of epochs with no improvement after which learning rate will be reduced.
        min_lr: Lower bound on learning rate.
        verbose: If True, logs a message to stdout on each update.

    Returns:
        LRScheduler or None.
    """
    if scheduler_type.lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        logger.info(
            f"Initialized ReduceLROnPlateau scheduler (mode={mode}, factor={factor}, patience={patience}, min_lr={min_lr})"
        )
        return scheduler
    elif scheduler_type.lower() in ("none", "", "null"):
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def get_current_lr(optimizer: Optimizer) -> float:
    """Returns the current learning rate from the first parameter group."""
    return float(optimizer.param_groups[0]["lr"])
