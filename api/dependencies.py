"""FastAPI dependencies and model lifecycle management for LipRead.

Provides singleton management of the InferenceEngine so weights are loaded
once into memory and reused across requests rather than re-initialized per call.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from fastapi import Depends, HTTPException, status

from src.inference.engine import InferenceEngine
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.api.dependencies")

# Default checkpoint search paths in priority order
DEFAULT_CHECKPOINT_PATHS = [
    Path("E:/LipRead_Data/checkpoints/best_model.pt"),
    Path("E:/LipRead_Data/checkpoints/latest_model.pt"),
    Path("checkpoints/best_model.pt"),
    Path("checkpoints/latest_model.pt"),
]


def resolve_checkpoint_path(custom_path: Optional[str | Path] = None) -> Optional[Path]:
    """Resolves the active checkpoint path from argument, env var, or default locations."""
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p.resolve()
        logger.warning("Custom checkpoint path does not exist: %s", custom_path)
        return None

    # Check environment variable
    env_path = os.environ.get("LIPREAD_CHECKPOINT_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists() and p.is_file():
            return p.resolve()
        logger.warning("LIPREAD_CHECKPOINT_PATH points to non-existent file: %s", env_path)

    # Check known filesystem paths
    for p in DEFAULT_CHECKPOINT_PATHS:
        if p.exists() and p.is_file():
            return p.resolve()

    return None


class ModelManager:
    """Manages the lifecycle and singleton instance of the LipRead InferenceEngine."""

    def __init__(self) -> None:
        self._engine: Optional[InferenceEngine] = None
        self._checkpoint_path: Optional[Path] = None
        self._device: Optional[str] = None
        self._error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Returns True if the InferenceEngine is active and ready."""
        return self._engine is not None

    @property
    def engine(self) -> Optional[InferenceEngine]:
        """Returns the loaded InferenceEngine or None."""
        return self._engine

    @property
    def checkpoint_path(self) -> Optional[Path]:
        """Returns the active checkpoint path."""
        return self._checkpoint_path

    @property
    def checkpoint_name(self) -> Optional[str]:
        """Returns the filename of the loaded checkpoint."""
        return self._checkpoint_path.name if self._checkpoint_path else None

    @property
    def device_name(self) -> Optional[str]:
        """Returns the compute device name (e.g., 'cuda:0' or 'cpu')."""
        return self._device

    @property
    def error(self) -> Optional[str]:
        """Returns the last loading error message if failed, else None."""
        return self._error

    def load(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Loads or reloads the InferenceEngine from a checkpoint file.

        Args:
            checkpoint_path: Optional custom path to a .pt checkpoint.
            device: Target device string or torch.device.

        Returns:
            Tuple of (success: bool, error_message: Optional[str]).
        """
        resolved = resolve_checkpoint_path(checkpoint_path)
        if resolved is None:
            self._error = (
                f"No checkpoint found. Checked: {checkpoint_path or [str(p) for p in DEFAULT_CHECKPOINT_PATHS]}"
            )
            logger.error("ModelManager failed to load: %s", self._error)
            self._engine = None
            self._checkpoint_path = None
            return False, self._error

        try:
            target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Initializing InferenceEngine from checkpoint: %s on %s", resolved, target_device)
            self._engine = InferenceEngine.from_checkpoint(
                checkpoint_path=resolved,
                device=target_device,
            )
            self._checkpoint_path = resolved
            self._device = str(self._engine.device)
            self._error = None
            logger.info("InferenceEngine successfully loaded and cached.")
            return True, None
        except Exception as exc:
            self._error = f"Failed to initialize InferenceEngine: {exc}"
            logger.exception("Model loading encountered an exception")
            self._engine = None
            self._checkpoint_path = None
            return False, self._error

    def unload(self) -> None:
        """Unloads the active model from memory."""
        self._engine = None
        self._checkpoint_path = None
        self._device = None
        self._error = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("InferenceEngine unloaded.")


# Global singleton instance for application lifespan
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """FastAPI dependency provider for the ModelManager singleton."""
    return model_manager


def get_inference_engine(
    manager: ModelManager = Depends(get_model_manager),
) -> InferenceEngine:
    """FastAPI dependency providing the active InferenceEngine.

    Raises:
        HTTPException(503): If the model is not loaded or checkpoint is unavailable.
    """
    if not manager.is_loaded or manager.engine is None:
        error_detail = manager.error or "Model checkpoint is not loaded."
        logger.warning("Inference requested but model is unavailable: %s", error_detail)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {error_detail}",
        )
    return manager.engine


# Global singleton instance for GenAI contextual post-processing
_corrector_singleton = None


def get_contextual_corrector():
    """FastAPI dependency providing the active ContextualCorrector instance."""
    global _corrector_singleton
    if _corrector_singleton is None:
        from src.genai import get_corrector

        _corrector_singleton = get_corrector()
    return _corrector_singleton

