"""Connectionist Temporal Classification (CTC) loss wrapper and decoding utilities for LipRead.

Provides:
  - CTCLossWrapper: Configured PyTorch CTCLoss handler supporting both 2D padded and 1D flattened targets
  - greedy_decode: Basic argmax CTC collapse utility removing consecutive duplicates and blank tokens
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset_config import VocabularyConfig
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.models.ctc")


class CTCLossWrapper(nn.Module):
    """Wrapper around PyTorch nn.CTCLoss with dimension handling and validation."""

    def __init__(
        self,
        blank: int = 0,
        zero_infinity: bool = True,
        reduction: str = "mean",
    ):
        """Initializes the CTC Loss wrapper.

        Args:
            blank: CTC blank symbol index (default: 0).
            zero_infinity: Whether to zero infinite losses when inputs are shorter than targets (default: True).
            reduction: Reduction mode ('mean', 'sum', 'none') (default: 'mean').
        """
        super().__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.reduction = reduction

        self.loss_fn = nn.CTCLoss(
            blank=blank,
            zero_infinity=zero_infinity,
            reduction=reduction,
        )

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the Connectionist Temporal Classification loss.

        Args:
            log_probs: Model log-probabilities of shape [T, B, V] (or [B, T, V] which is transposed).
            targets: Target character IDs tensor of shape [B, max_L] or 1D flattened [sum(target_lengths)].
            input_lengths: Sequence lengths of inputs [B].
            target_lengths: Lengths of target sequences [B].

        Returns:
            Computed scalar CTC loss tensor (or tensor of shape [B] if reduction='none').

        Raises:
            ValueError: If tensor shapes or lengths are inconsistent.
        """
        # Ensure log_probs is in [T, B, V] layout required by PyTorch CTCLoss
        if log_probs.ndim != 3:
            raise ValueError(
                f"Expected 3D log_probs tensor [T, B, V] or [B, T, V], got ndim={log_probs.ndim} "
                f"(shape: {tuple(log_probs.shape)})"
            )

        batch_size = input_lengths.shape[0]

        # If log_probs is [B, T, V] instead of [T, B, V], transpose it
        if log_probs.shape[0] == batch_size and log_probs.shape[1] != batch_size:
            log_probs = log_probs.transpose(0, 1)

        # PyTorch CTCLoss expects input_lengths and target_lengths as 1D CPU/CUDA integer tensors
        if input_lengths.ndim != 1:
            input_lengths = input_lengths.view(-1)
        if target_lengths.ndim != 1:
            target_lengths = target_lengths.view(-1)

        # Ensure valid log probabilities over vocabulary dimension
        log_probs = F.log_softmax(log_probs, dim=-1)

        # Compute CTC loss
        loss = self.loss_fn(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        return loss


def greedy_decode(
    logits: torch.Tensor,
    blank_idx: int = 0,
    vocab: Optional[VocabularyConfig] = None,
) -> List[List[int]]:
    """Performs best-path greedy CTC decoding on model outputs.

    Takes argmax at each timestep, collapses consecutive identical tokens,
    and discards blank tokens.

    Args:
        logits: Logits or probabilities of shape [B, T, V] or [T, B, V].
        blank_idx: Index of CTC blank token (default: 0).
        vocab: Optional VocabularyConfig instance for character mapping.

    Returns:
        List of decoded token ID lists, one per batch item: List[List[int]].
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D tensor for CTC decoding, got ndim={logits.ndim}")

    # Determine layout: standardise to [B, T, V]
    if logits.shape[0] > logits.shape[1] and logits.shape[1] <= 128:
        # Likely [T, B, V] layout; transpose to [B, T, V]
        x = logits.transpose(0, 1)
    else:
        x = logits

    # Argmax over vocabulary dimension V
    preds = torch.argmax(x, dim=-1)  # Shape: [B, T]

    batch_decoded: List[List[int]] = []
    for b in range(preds.shape[0]):
        seq = preds[b].tolist()
        collapsed: List[int] = []
        prev_token = -1

        for token in seq:
            # Only add token if it differs from immediate previous (collapse repeats)
            # and is not the blank token
            if token != prev_token:
                if token != blank_idx:
                    collapsed.append(token)
                prev_token = token

        batch_decoded.append(collapsed)

    return batch_decoded


def decode_to_text(
    logits: torch.Tensor,
    blank_idx: int = 0,
    vocab: Optional[VocabularyConfig] = None,
) -> List[str]:
    """Greedy decodes model outputs into text strings using VocabularyConfig."""
    vocab_cfg = vocab or VocabularyConfig()
    token_lists = greedy_decode(logits, blank_idx=blank_idx, vocab=vocab_cfg)
    return [vocab_cfg.decode(tokens) for tokens in token_lists]
