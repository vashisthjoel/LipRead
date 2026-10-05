"""Production training engine for LipRead.

Orchestrates the complete model training lifecycle:
- Forward pass through 3D CNN + Bi-LSTM
- CTC Loss computation with PyTorch CTCLoss
- Gradient accumulation and gradient norm clipping
- Mixed precision training with torch.amp.autocast and GradScaler
- Model checkpointing (best_model.pt and latest_model.pt)
- Evaluation and error rate metrics (CER / WER)
- Local MLflow experiment tracking at E:\\LipRead_Data\\mlruns
- CUDA memory tracking and safety checks
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data.dataset_config import VocabularyConfig
from src.data.paths import get_checkpoint_dir, get_mlruns_dir
from src.models.ctc import CTCLossWrapper
from src.models.lipread_model import LipReadModel
from src.training.checkpoint import save_checkpoint
from src.training.scheduler import get_current_lr
from src.training.validation import ValidationResult, evaluate

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-end Trainer for LipReadModel on EGCLLC."""

    def __init__(
        self,
        model: LipReadModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        vocab: Optional[VocabularyConfig] = None,
        device: Optional[torch.device | str] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 2,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: Optional[Path] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "LipRead",
        run_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vocab = vocab or VocabularyConfig()
        self.config_dict = config_dict or {}
        self.seed = seed
        self.run_name = run_name

        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else get_checkpoint_dir()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.ctc_loss_fn = CTCLossWrapper(blank=self.vocab.blank_idx, zero_infinity=True)

        # Mixed Precision (AMP)
        self.is_cuda = self.device.type == "cuda"
        self.use_amp = use_amp and self.is_cuda
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # MLflow setup
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or str(get_mlruns_dir().as_uri())
        self.mlflow_run = None
        self._init_mlflow()

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.invalid_samples_count = 0

        logger.info(
            f"Trainer initialized: device={self.device}, AMP={self.use_amp}, "
            f"grad_accum={self.gradient_accumulation_steps}, clip_norm={self.gradient_clip_norm}"
        )

    def _init_mlflow(self) -> None:
        """Initializes local MLflow tracking without breaking if MLflow is uninstalled."""
        try:
            import os
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_available = True
            logger.info(f"MLflow initialized at: {self.tracking_uri} (Experiment: {self.experiment_name})")
        except ImportError:
            self.mlflow_available = False
            logger.warning("MLflow package not available; running without experiment tracking.")
        except Exception as e:
            self.mlflow_available = False
            logger.warning(f"Could not initialize MLflow: {e}")

    def log_mlflow_params(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters to active MLflow run."""
        if self.mlflow_available and self.mlflow_run:
            try:
                import mlflow
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log MLflow params: {e}")

    def log_mlflow_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs scalar metrics to active MLflow run."""
        if self.mlflow_available and self.mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log MLflow metrics: {e}")

    def train_epoch(
        self,
        epoch: int,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Trains the model for one complete epoch (or up to max_batches).

        Args:
            epoch: Current 1-based epoch index.
            max_batches: Optional limit on batches processed.

        Returns:
            Dictionary with training statistics (train_loss, throughput, memory, etc.).
        """
        self.model.train()
        self.current_epoch = epoch
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = 0
        total_samples = 0
        epoch_start_time = time.perf_counter()

        vram_before = torch.cuda.memory_allocated(self.device) / (1024**2) if self.is_cuda else 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                video = batch["frames"].to(self.device, non_blocking=True)
                targets = batch["tokens"].to(self.device, non_blocking=True)
                input_lengths = batch["input_lengths"].to(self.device, non_blocking=True)
                target_lengths = batch["target_lengths"].to(self.device, non_blocking=True)
            else:
                video = getattr(batch, "video", getattr(batch, "frames")).to(self.device, non_blocking=True)
                targets = getattr(batch, "targets", getattr(batch, "tokens")).to(self.device, non_blocking=True)
                input_lengths = batch.input_lengths.to(self.device, non_blocking=True)
                target_lengths = batch.target_lengths.to(self.device, non_blocking=True)

            batch_size = video.size(0)
            total_samples += batch_size

            # Forward pass under AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(video)  # [B, T, V]
                ctc_logits = self.model.to_ctc_layout(logits)  # [T, B, V]
                loss = self.ctc_loss_fn(ctc_logits, targets, input_lengths, target_lengths)

            # Check for non-finite loss
            loss_item = loss.item()
            if not math.isfinite(loss_item):
                self.invalid_samples_count += 1
                logger.warning(
                    f"[Batch {batch_idx}] Non-finite CTC loss: {loss_item}. "
                    f"Target lengths: {target_lengths.tolist()}, Input lengths: {input_lengths.tolist()}."
                )
                continue

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()

            if num_batches == 0:
                initial_loss = loss_item
            final_loss = loss_item

            total_loss += loss_item
            num_batches += 1
            self.global_step += 1

            # Step optimizer upon reaching accumulation threshold or end of dataset
            is_accum_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader) or (
                max_batches is not None and (batch_idx + 1) == max_batches
            )

            if is_accum_step or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        elapsed = time.perf_counter() - epoch_start_time
        avg_loss = total_loss / max(num_batches, 1)
        samples_per_sec = total_samples / max(elapsed, 1e-4)
        batches_per_sec = num_batches / max(elapsed, 1e-4)

        vram_after = torch.cuda.memory_allocated(self.device) / (1024**2) if self.is_cuda else 0.0
        vram_peak = torch.cuda.max_memory_allocated(self.device) / (1024**2) if self.is_cuda else 0.0

        stats = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "initial_loss": initial_loss if num_batches > 0 else 0.0,
            "final_loss": final_loss if num_batches > 0 else 0.0,
            "batches_processed": num_batches,
            "samples_processed": total_samples,
            "elapsed_sec": elapsed,
            "samples_per_sec": samples_per_sec,
            "batches_per_sec": batches_per_sec,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "vram_peak_mb": vram_peak,
            "learning_rate": get_current_lr(self.optimizer),
            "invalid_samples": self.invalid_samples_count,
        }

        return stats

    def fit(
        self,
        epochs: int = 5,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the complete multi-epoch training and validation loop.

        Args:
            epochs: Total epochs to run.
            max_train_batches: Optional cap on training batches per epoch.
            max_val_batches: Optional cap on validation batches per epoch.

        Returns:
            Dictionary containing history, best metrics, and checkpoint paths.
        """
        if self.mlflow_available:
            try:
                import mlflow
                run_title = self.run_name or f"run_epochs_{epochs}_batch_{self.train_loader.batch_size}"
                self.mlflow_run = mlflow.start_run(run_name=run_title)
                # Log model & train params
                train_samples = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else None
                val_samples = len(self.val_loader.dataset) if (self.val_loader and hasattr(self.val_loader, "dataset")) else 0
                total_params = sum(p.numel() for p in self.model.parameters())
                gpu_name = torch.cuda.get_device_name(self.device) if self.is_cuda else "CPU"

                mlflow_params = {
                    "epochs": epochs,
                    "batch_size": self.train_loader.batch_size,
                    "effective_batch_size": (self.train_loader.batch_size or 2) * self.gradient_accumulation_steps,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "learning_rate": get_current_lr(self.optimizer),
                    "weight_decay": self.optimizer.defaults.get("weight_decay", 1e-4),
                    "optimizer": self.optimizer.__class__.__name__,
                    "gradient_clip_norm": self.gradient_clip_norm,
                    "use_amp": self.use_amp,
                    "device": str(self.device),
                    "gpu": gpu_name,
                    "model_name": getattr(self.model, "name", "lipread_3dcnn_bilstm"),
                    "model_parameters": total_params,
                    "vocab_size": self.vocab.vocab_size,
                    "seed": self.seed,
                    "train_sample_count": train_samples,
                    "validation_sample_count": val_samples,
                }
                mlflow.log_params(mlflow_params)
            except Exception as e:
                logger.warning(f"Could not start MLflow run: {e}")

        history: List[Dict[str, Any]] = []
        best_checkpoint_path: Optional[Path] = None
        latest_checkpoint_path: Optional[Path] = None

        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")

        try:
            for epoch in range(1, epochs + 1):
                # 1. Train epoch
                train_stats = self.train_epoch(epoch, max_batches=max_train_batches)

                # 2. Validation
                val_result: Optional[ValidationResult] = None
                if self.val_loader is not None:
                    val_result = evaluate(
                        model=self.model,
                        dataloader=self.val_loader,
                        ctc_loss_fn=self.ctc_loss_fn,
                        device=self.device,
                        vocab=self.vocab,
                        max_batches=max_val_batches,
                    )

                val_loss = val_result.val_loss if val_result is not None else float("inf")
                cer = val_result.cer if val_result is not None else 1.0
                wer = val_result.wer if val_result is not None else 1.0

                # 3. Learning rate scheduler step
                if self.scheduler is not None:
                    if hasattr(self.scheduler, "step"):
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()

                # 4. Checkpoint saving
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                checkpoint_state = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                    "best_validation_loss": self.best_val_loss,
                    "train_loss": train_stats["train_loss"],
                    "val_loss": val_loss,
                    "cer": cer,
                    "wer": wer,
                    "config": self.config_dict,
                    "random_seed": self.seed,
                }

                latest_path, best_path = save_checkpoint(
                    state=checkpoint_state,
                    is_best=is_best,
                    checkpoint_dir=self.checkpoint_dir,
                )
                latest_checkpoint_path = latest_path
                if best_path is not None:
                    best_checkpoint_path = best_path

                # 5. Log metrics
                epoch_metrics = {
                    "train_loss": train_stats["train_loss"],
                    "val_loss": val_loss,
                    "cer": cer,
                    "wer": wer,
                    "learning_rate": train_stats["learning_rate"],
                    "epoch_time_sec": train_stats["elapsed_sec"],
                    "best_val_loss": self.best_val_loss,
                    "vram_peak_mb": train_stats["vram_peak_mb"],
                }
                self.log_mlflow_metrics(epoch_metrics, step=epoch)

                epoch_summary = {
                    **train_stats,
                    "val_loss": val_loss,
                    "cer": cer,
                    "wer": wer,
                    "is_best": is_best,
                }
                history.append(epoch_summary)

                # Log console summary
                logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_stats['train_loss']:.4f} | "
                    f"Val Loss: {val_loss:.4f} | CER: {cer:.4f} | WER: {wer:.4f} | "
                    f"LR: {train_stats['learning_rate']:.2e} | Time: {train_stats['elapsed_sec']:.1f}s | "
                    f"Peak VRAM: {train_stats['vram_peak_mb']:.1f}MB"
                )

                if val_result and val_result.sample_predictions:
                    for i, sample in enumerate(val_result.sample_predictions[:2]):
                        logger.info(
                            f"  [Sample {i+1}] Ref: '{sample['reference']}' | Pred: '{sample['prediction']}'"
                        )

        finally:
            if self.mlflow_available and self.mlflow_run:
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception:
                    pass

        run_id = self.mlflow_run.info.run_id if (self.mlflow_available and self.mlflow_run) else "none"

        return {
            "history": history,
            "best_val_loss": self.best_val_loss,
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
            "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path else None,
            "mlflow_run_id": run_id,
            "mlflow_experiment": self.experiment_name,
        }
