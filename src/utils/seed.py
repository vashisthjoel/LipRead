"""Reproducibility utilities for LipRead.

Provides deterministic seeding across Python, NumPy, PyTorch, and CUDA.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_cuda: bool = False) -> int:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: The integer random seed (default 42).
        deterministic_cuda: If True, enables PyTorch deterministic algorithms
            (torch.backends.cudnn.deterministic = True, benchmark = False).
            Note: full deterministic CUDA behavior may reduce runtime performance
            on certain convolution operations on GTX 1650. Default is False.

    Returns:
        The seed value that was set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Deterministic CUDA enabled (may slightly reduce throughput).")
        else:
            torch.backends.cudnn.benchmark = True

    logger.debug(f"Global random seed set to {seed} (deterministic_cuda={deterministic_cuda}).")
    return seed
