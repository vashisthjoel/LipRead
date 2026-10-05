"""Custom PyTorch collate function for LipRead.

Pads variable-length target token sequences and stacks fixed-length
spatiotemporal video tensors into batches compatible with PyTorch CTC loss.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of dataset samples into a padded batch dictionary.

    Args:
        batch: List of sample dictionaries produced by EGCLLCDataset.

    Returns:
        Dictionary containing:
          - 'frames': FloatTensor of shape [B, T, C, H, W]
          - 'tokens': LongTensor of shape [B, max_target_len] (padded with 0)
          - 'input_lengths': LongTensor of shape [B]
          - 'target_lengths': LongTensor of shape [B]
          - 'transcripts': List[str] of original text sentences
          - 'sample_ids': List[str]
          - 'speaker_ids': List[str]
          - 'video_ids': List[str]
    """
    if not batch:
        raise ValueError("Empty batch passed to pad_collate_fn.")

    # 1. Stack video tensors: [B, T, C, H, W]
    frames_batch = torch.stack([b["frames"] for b in batch], dim=0)
    batch_size = len(batch)

    # 2. Extract input lengths for CTC loss (T dimension)
    input_lengths = torch.tensor([b["frames"].shape[0] for b in batch], dtype=torch.long)

    # 3. Extract and pad variable-length target token sequences
    target_lengths = torch.tensor([b["tokens"].numel() for b in batch], dtype=torch.long)
    max_target_len = int(target_lengths.max().item()) if batch_size > 0 else 0

    tokens_batch = torch.zeros((batch_size, max_target_len), dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["tokens"]
        if t.numel() > 0:
            tokens_batch[i, : t.numel()] = t

    # 4. Gather text and identifiers
    transcripts = [b["transcript"] for b in batch]
    sample_ids = [b["sample_id"] for b in batch]
    speaker_ids = [b["speaker_id"] for b in batch]
    video_ids = [b["video_id"] for b in batch]

    return {
        "frames": frames_batch,
        "tokens": tokens_batch,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "transcripts": transcripts,
        "sample_ids": sample_ids,
        "speaker_ids": speaker_ids,
        "video_ids": video_ids,
    }
