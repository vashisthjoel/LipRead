"""Model utility functions for inspection, parameter counting, device movement, and layout helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.models.model_utils")


def count_parameters(model: nn.Module) -> int:
    """Calculates the total number of parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Calculates the number of trainable (requires_grad=True) parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Generates a structured dictionary summary of model architecture and parameter count.

    Args:
        model: PyTorch module.

    Returns:
        Dictionary containing parameter counts and layer details.
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    submodules = {}
    for name, child in model.named_children():
        submodules[name] = {
            "type": child.__class__.__name__,
            "total_params": count_parameters(child),
            "trainable_params": count_trainable_parameters(child),
        }

    return {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_size_mb": round((total_params * 4) / (1024 * 1024), 2),  # float32 = 4 bytes
        "components": submodules,
    }


def to_device(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, torch.device]:
    """Moves model to target device (or automatically detects CUDA if available).

    Args:
        model: PyTorch module.
        device: Target device string or torch.device. If None, selects CUDA if available else CPU.

    Returns:
        Tuple of (model moved to device, target torch.device).
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model = model.to(target_device)
    logger.debug("Moved %s to device: %s", model.__class__.__name__, target_device)
    return model, target_device


def to_ctc_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a [B, T, V] tensor into PyTorch CTC [T, B, V] layout.

    Args:
        tensor: 3D tensor of shape [B, T, V].

    Returns:
        Tensor view of shape [T, B, V].
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor [B, T, V], got ndim={tensor.ndim}")
    return tensor.transpose(0, 1)


def to_log_probs(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Computes log-softmax probabilities from unnormalized logits."""
    return F.log_softmax(logits, dim=dim)
