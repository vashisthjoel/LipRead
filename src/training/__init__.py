"""Training package for LipRead.

Provides trainer engine, validation evaluators, metrics, checkpoint managers,
and learning rate schedulers for 3D CNN + Bi-LSTM visual speech recognition.
"""

from src.training.checkpoint import get_default_checkpoint_dir, load_checkpoint, save_checkpoint
from src.training.metrics import (
    compute_batch_cer,
    compute_batch_wer,
    compute_cer,
    compute_edit_distance,
    compute_wer,
)
from src.training.scheduler import build_optimizer, build_scheduler, get_current_lr
from src.training.trainer import Trainer
from src.training.validation import ValidationResult, evaluate

__all__ = [
    "Trainer",
    "evaluate",
    "ValidationResult",
    "compute_edit_distance",
    "compute_cer",
    "compute_wer",
    "compute_batch_cer",
    "compute_batch_wer",
    "save_checkpoint",
    "load_checkpoint",
    "get_default_checkpoint_dir",
    "build_optimizer",
    "build_scheduler",
    "get_current_lr",
]
