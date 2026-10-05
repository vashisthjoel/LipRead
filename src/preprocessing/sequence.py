"""Temporal sequence builder and dimension formatting for LipRead.

Produces fixed-length temporal tensors with explicit shape convention:
    [T, C, H, W]
where:
    T = Sequence length (default: 75 frames, corresponding to ~3.0s at 25 fps)
    C = Channel depth (default: 1 for grayscale)
    H = Frame height in pixels (default: 88)
    W = Frame width in pixels (default: 88)

Target output shape: (75, 1, 88, 88)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

from src.utils.logging_config import get_logger

logger = get_logger("lipread.preprocessing.sequence")


class SequenceBuilder:
    """Standardizes arbitrary-length frame lists into fixed temporal tensors [T, C, H, W]."""

    def __init__(
        self,
        sequence_length: int = 75,
        image_size: int = 88,
        channels: int = 1,
        padding_strategy: str = "edge",
        truncation_strategy: str = "center",
    ):
        """Initializes the SequenceBuilder.

        Args:
            sequence_length: Target number of temporal frames (T). Default: 75.
            image_size: Target height and width in pixels (H, W). Default: 88.
            channels: Channel count (C). Default: 1 (grayscale).
            padding_strategy: Strategy for handling sequences shorter than sequence_length.
                - 'edge': Replicates the final frame to preserve speaker pose.
                - 'zero': Appends zero-valued frames.
            truncation_strategy: Strategy for handling sequences longer than sequence_length.
                - 'center': Slices a symmetric temporal window around the middle.
                - 'start': Takes the first T frames.
                - 'end': Takes the last T frames.
        """
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.channels = channels
        self.padding_strategy = padding_strategy.lower()
        self.truncation_strategy = truncation_strategy.lower()

        if self.padding_strategy not in ("edge", "zero"):
            raise ValueError(f"Unknown padding_strategy: {padding_strategy}. Must be 'edge' or 'zero'.")

        if self.truncation_strategy not in ("center", "start", "end"):
            raise ValueError(
                f"Unknown truncation_strategy: {truncation_strategy}. Must be 'center', 'start', or 'end'."
            )

    def _pad_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Pads a sequence of length < target_length up to target_length."""
        current_len = len(frames)
        diff = self.sequence_length - current_len

        if current_len == 0:
            # Entirely empty input: fill with zero frames
            zero_frame = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            return [zero_frame] * self.sequence_length

        if self.padding_strategy == "edge":
            # Replicate the terminal frame
            last_frame = frames[-1]
            return frames + [last_frame.copy() for _ in range(diff)]
        elif self.padding_strategy == "zero":
            zero_frame = np.zeros_like(frames[0])
            return frames + [zero_frame.copy() for _ in range(diff)]

        return frames

    def _truncate_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Truncates a sequence of length > target_length down to target_length."""
        total_len = len(frames)
        t = self.sequence_length

        if self.truncation_strategy == "center":
            start_idx = (total_len - t) // 2
            return frames[start_idx : start_idx + t]
        elif self.truncation_strategy == "start":
            return frames[:t]
        elif self.truncation_strategy == "end":
            return frames[-t:]

        return frames[:t]

    def build_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """Constructs a normalized NumPy sequence array with shape [T, C, H, W].

        Args:
            frames: List of 2D (H, W) or 3D (H, W, C) NumPy arrays in temporal order.

        Returns:
            NumPy array of shape (75, 1, 88, 88) with float32 dtype.
        """
        if not frames:
            logger.warning("Empty frame list passed to build_sequence. Generating zero tensor.")
            return np.zeros(
                (self.sequence_length, self.channels, self.image_size, self.image_size),
                dtype=np.float32,
            )

        # Apply temporal alignment
        if len(frames) < self.sequence_length:
            aligned_frames = self._pad_sequence(frames)
        elif len(frames) > self.sequence_length:
            aligned_frames = self._truncate_sequence(frames)
        else:
            aligned_frames = frames

        # Standardize individual frame dimensions to (C, H, W)
        formatted_frames: List[np.ndarray] = []
        for f in aligned_frames:
            f_arr = np.asarray(f, dtype=np.float32)
            if f_arr.ndim == 2:
                # Shape (H, W) -> add channel axis (1, H, W)
                f_ch = np.expand_dims(f_arr, axis=0)
            elif f_arr.ndim == 3:
                if f_arr.shape[2] == self.channels:
                    # Shape (H, W, C) -> transpose to (C, H, W)
                    f_ch = np.transpose(f_arr, (2, 0, 1))
                elif f_arr.shape[0] == self.channels:
                    # Already (C, H, W)
                    f_ch = f_arr
                else:
                    raise ValueError(f"Unexpected frame channel layout: {f_arr.shape}")
            else:
                raise ValueError(f"Frame has invalid number of dimensions ({f_arr.ndim}): {f_arr.shape}")

            # Verify spatial dimensions match target
            if f_ch.shape[1:] != (self.image_size, self.image_size):
                raise ValueError(
                    f"Frame spatial size {f_ch.shape[1:]} does not match target ({self.image_size}, {self.image_size})"
                )

            formatted_frames.append(f_ch)

        # Stack across temporal dimension: result shape is (T, C, H, W)
        sequence_np = np.stack(formatted_frames, axis=0)
        expected_shape = (self.sequence_length, self.channels, self.image_size, self.image_size)

        if sequence_np.shape != expected_shape:
            raise RuntimeError(
                f"Generated sequence shape {sequence_np.shape} does not match expected {expected_shape}"
            )

        return sequence_np

    def to_tensor(self, sequence_np: np.ndarray) -> torch.Tensor:
        """Converts sequence NumPy array to a PyTorch float32 Tensor.

        Args:
            sequence_np: NumPy array of shape (T, C, H, W).

        Returns:
            PyTorch Tensor of shape [T, C, H, W] on CPU.
        """
        tensor = torch.from_numpy(sequence_np).float()
        return tensor
