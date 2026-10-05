"""Validation logic and evaluation runner for LipRead.

Evaluates model performance on validation / test splits:
- Computes mean CTC validation loss
- Decodes predictions greedily
- Computes Character Error Rate (CER) and Word Error Rate (WER)
- Extracts sample qualitative pairs for inspection
- Never updates model gradients
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset_config import VocabularyConfig
from src.models.ctc import CTCLossWrapper, decode_to_text
from src.training.metrics import compute_batch_cer, compute_batch_wer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Encapsulates validation performance metrics and qualitative predictions."""

    val_loss: float
    cer: float
    wer: float
    num_samples: int
    num_batches: int
    elapsed_sec: float
    samples_per_sec: float
    sample_predictions: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "val_loss": self.val_loss,
            "cer": self.cer,
            "wer": self.wer,
            "num_samples": self.num_samples,
            "num_batches": self.num_batches,
            "elapsed_sec": self.elapsed_sec,
            "samples_per_sec": self.samples_per_sec,
        }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    ctc_loss_fn: CTCLossWrapper,
    device: torch.device | str,
    vocab: VocabularyConfig,
    max_batches: Optional[int] = None,
    max_sample_pairs_to_log: int = 3,
) -> ValidationResult:
    """Runs a complete evaluation pass over a DataLoader without gradient tracking.

    Args:
        model: The LipReadModel instance.
        dataloader: DataLoader yielding BatchedSample batches.
        ctc_loss_fn: CTCLossWrapper instance.
        device: Target execution device ('cuda' or 'cpu').
        vocab: Vocabulary configuration for string decoding.
        max_batches: Optional upper bound on batches to evaluate (useful for smoke tests).
        max_sample_pairs_to_log: Maximum number of sample predictions to collect.

    Returns:
        ValidationResult with loss, CER, WER, and sample transcripts.
    """
    model.eval()
    total_loss = 0.0
    num_batches_evaluated = 0
    total_samples = 0

    all_references: List[str] = []
    all_hypotheses: List[str] = []
    qualitative_samples: List[Dict[str, str]] = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                video = batch["frames"].to(device, non_blocking=True)
                targets = batch["tokens"].to(device, non_blocking=True)
                input_lengths = batch["input_lengths"].to(device, non_blocking=True)
                target_lengths = batch["target_lengths"].to(device, non_blocking=True)
                raw_transcripts = batch["transcripts"]
                sample_ids = batch.get("sample_ids", [f"sample_{i}" for i in range(len(raw_transcripts))])
            else:
                video = getattr(batch, "video", getattr(batch, "frames")).to(device, non_blocking=True)
                targets = getattr(batch, "targets", getattr(batch, "tokens")).to(device, non_blocking=True)
                input_lengths = batch.input_lengths.to(device, non_blocking=True)
                target_lengths = batch.target_lengths.to(device, non_blocking=True)
                raw_transcripts = batch.transcripts
                sample_ids = getattr(batch, "sample_ids", [f"sample_{i}" for i in range(len(raw_transcripts))])

            batch_size = video.size(0)
            total_samples += batch_size

            # Forward pass
            logits = model(video)  # [B, T, V]
            ctc_logits = model.to_ctc_layout(logits)  # [T, B, V]

            # Calculate CTC loss
            loss = ctc_loss_fn(ctc_logits, targets, input_lengths, target_lengths)
            loss_val = loss.item()
            total_loss += loss_val
            num_batches_evaluated += 1

            # Decode predictions
            hypotheses = decode_to_text(logits, blank_idx=vocab.blank_idx, vocab=vocab)
            references = [t.strip() for t in raw_transcripts]

            all_references.extend(references)
            all_hypotheses.extend(hypotheses)

            # Collect a few sample predictions for inspection
            if len(qualitative_samples) < max_sample_pairs_to_log:
                for sid, ref, hyp in zip(sample_ids, references, hypotheses):
                    if len(qualitative_samples) < max_sample_pairs_to_log:
                        qualitative_samples.append({
                            "sample_id": sid,
                            "reference": ref,
                            "prediction": hyp,
                        })

    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(num_batches_evaluated, 1)
    samples_per_sec = total_samples / max(elapsed, 1e-4)

    cer = compute_batch_cer(all_references, all_hypotheses) if all_references else 1.0
    wer = compute_batch_wer(all_references, all_hypotheses) if all_references else 1.0

    return ValidationResult(
        val_loss=avg_loss,
        cer=cer,
        wer=wer,
        num_samples=total_samples,
        num_batches=num_batches_evaluated,
        elapsed_sec=elapsed,
        samples_per_sec=samples_per_sec,
        sample_predictions=qualitative_samples,
    )
