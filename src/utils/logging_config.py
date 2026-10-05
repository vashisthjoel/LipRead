"""Application logging configuration for LipRead.

Provides standard formatting, console handlers, and optional file logging
across all modules.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "lipread",
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a standard application logger.

    Args:
        name: Name of the logger (defaults to "lipread").
        level: Logging level (e.g. "DEBUG", "INFO", "WARNING", "ERROR").
               Defaults to value of LOG_LEVEL environment variable or "INFO".
        log_file: Optional path to output log file.

    Returns:
        Configured logging.Logger instance.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if logger was already initialized
    if not logger.handlers:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "lipread") -> logging.Logger:
    """Convenience helper to retrieve an existing or newly configured logger."""
    return setup_logger(name=name)
