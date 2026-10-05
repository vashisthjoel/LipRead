"""Adapter transforming pre-cropped EGCLLC lip image sequences into standard [T, C, H, W] tensors.

EGCLLC already provides pre-cropped lip images. This adapter loads images lazily,
converts to grayscale, resizes to (88, 88), normalizes to [0.0, 1.0], and aligns
temporal sequence length to 75 frames using SequenceBuilder without invoking MediaPipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

from src.preprocessing.sequence import SequenceBuilder
from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger("lipread.data.egcllc_adapter")


class EGCLLCAdapter:
    """Transforms raw EGCLLC lip image frame sequences into model-ready tensors."""

    def __init__(
        self,
        image_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        grayscale: Optional[bool] = None,
        normalize: Optional[bool] = None,
        padding_strategy: str = "edge",
        truncation_strategy: str = "center",
    ):
        """Initializes the adapter with configurable target dimensions.

        Args:
            image_size: Spatial height and width (default: 88).
            sequence_length: Temporal frame count (default: 75).
            grayscale: Whether to convert input frames to single-channel grayscale (default: True).
            normalize: Whether to scale pixel values into [0.0, 1.0] (default: True).
            padding_strategy: Strategy when frames < sequence_length ('edge' or 'zero').
            truncation_strategy: Strategy when frames > sequence_length ('center', 'start', 'end').
        """
        cfg = get_config().to_dict()
        prep_cfg = cfg.get("preprocessing", {})
        ds_cfg = cfg.get("dataset", {})

        self.image_size = (
            image_size
            if image_size is not None
            else ds_cfg.get("image_height", prep_cfg.get("image_size", 88))
        )
        self.sequence_length = (
            sequence_length
            if sequence_length is not None
            else ds_cfg.get("sequence_length", prep_cfg.get("sequence_length", 75))
        )
        self.grayscale = (
            grayscale
            if grayscale is not None
            else ds_cfg.get("grayscale", prep_cfg.get("grayscale", True))
        )
        self.normalize = (
            normalize
            if normalize is not None
            else ds_cfg.get("normalize", prep_cfg.get("normalization", True))
        )

        self.channels = 1 if self.grayscale else 3
        self.sequence_builder = SequenceBuilder(
            sequence_length=self.sequence_length,
            image_size=self.image_size,
            channels=self.channels,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
        )

    def load_frame(self, image_path: Union[str, Path]) -> np.ndarray:
        """Loads, converts, resizes, and normalizes a single image file.

        Args:
            image_path: Path to the image file (.jpg, .png).

        Returns:
            2D or 3D NumPy array of shape (image_size, image_size) or (image_size, image_size, C).
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Frame image not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Corrupted or unreadable image frame: {path}")

        # Grayscale conversion
        if self.grayscale and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Spatial resizing
        h, w = img.shape[:2]
        if (h, w) != (self.image_size, self.image_size):
            interp = cv2.INTER_AREA if (h >= self.image_size and w >= self.image_size) else cv2.INTER_LINEAR
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=interp)

        # Normalization to [0.0, 1.0]
        if self.normalize:
            return img.astype(np.float32) / 255.0
        return img

    def process_frame_paths(self, frame_paths: List[Union[str, Path]]) -> torch.Tensor:
        """Processes an ordered list of frame file paths into a [T, C, H, W] PyTorch tensor.

        Args:
            frame_paths: List of file paths to the sequential image frames.

        Returns:
            PyTorch float32 Tensor of shape [75, 1, 88, 88] (default configuration).
        """
        if not frame_paths:
            logger.warning("Empty frame path list provided. Returning blank zero tensor.")
            return torch.zeros(
                (self.sequence_length, self.channels, self.image_size, self.image_size),
                dtype=torch.float32,
            )

        processed_frames: List[np.ndarray] = []
        for p in frame_paths:
            try:
                frame = self.load_frame(p)
                processed_frames.append(frame)
            except Exception as exc:
                logger.debug("Failed reading frame %s: %s. Replicating previous frame.", p, exc)
                if processed_frames:
                    processed_frames.append(processed_frames[-1].copy())
                else:
                    processed_frames.append(
                        np.zeros((self.image_size, self.image_size), dtype=np.float32)
                    )

        # Standardize temporal dimension
        sequence_np = self.sequence_builder.build_sequence(processed_frames)
        return self.sequence_builder.to_tensor(sequence_np)

    def process_sequence_dir(self, sequence_dir: Union[str, Path]) -> torch.Tensor:
        """Convenience helper to process an entire video frame directory.

        Args:
            sequence_dir: Directory containing frame image files (.jpg, .png).

        Returns:
            PyTorch Tensor of shape [T, C, H, W].
        """
        dir_path = Path(sequence_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Sequence directory not found: {dir_path}")

        # Natural alphanumeric sorting of frame files (1.jpg, 2.jpg, ..., 10.jpg)
        frames = sorted(
            [f for f in dir_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem,
        )
        return self.process_frame_paths(frames)
