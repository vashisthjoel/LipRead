"""Deterministic Lip Region-of-Interest (ROI) extraction for LipRead.

Provides mouth bounding box calculation, aspect-ratio preserving square expansion,
padding, boundary clamping, cropping, grayscale conversion, 88x88 resizing,
and pixel normalization.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger("lipread.preprocessing.lip_roi")


def compute_mouth_bounding_box(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
    padding_ratio: float = 0.25,
    square: bool = True,
) -> Tuple[int, int, int, int]:
    """Computes a padded, boundary-clamped bounding box enclosing mouth landmarks.

    Args:
        landmarks: Array of shape (N, 2) containing (x, y) landmark pixel coordinates.
        frame_width: Width of the source frame in pixels.
        frame_height: Height of the source frame in pixels.
        padding_ratio: Relative padding applied around the mouth extent.
        square: If True, expands the bounding box to a 1:1 aspect ratio to avoid distortion.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) strictly clamped within frame boundaries.
    """
    if landmarks is None or len(landmarks) == 0:
        raise ValueError("Cannot compute bounding box from empty or None landmarks.")

    x_min = float(np.min(landmarks[:, 0]))
    x_max = float(np.max(landmarks[:, 0]))
    y_min = float(np.min(landmarks[:, 1]))
    y_max = float(np.max(landmarks[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    if square:
        # Enforce 1:1 square crop to preserve natural mouth geometry upon resize
        half_dim = (max(w, h) * (1.0 + 2.0 * padding_ratio)) / 2.0
        x1 = center_x - half_dim
        x2 = center_x + half_dim
        y1 = center_y - half_dim
        y2 = center_y + half_dim
    else:
        pad_x = w * padding_ratio
        pad_y = h * padding_ratio
        x1 = x_min - pad_x
        x2 = x_max + pad_x
        y1 = y_min - pad_y
        y2 = y_max + pad_y

    # Strict boundary clamping to prevent out-of-bounds indexing
    x1_clamped = int(max(0, min(frame_width - 1, round(x1))))
    y1_clamped = int(max(0, min(frame_height - 1, round(y1))))
    x2_clamped = int(max(x1_clamped + 1, min(frame_width, round(x2))))
    y2_clamped = int(max(y1_clamped + 1, min(frame_height, round(y2))))

    return x1_clamped, y1_clamped, x2_clamped, y2_clamped


def extract_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """Extracts a rectangular slice from a frame given a bounding box.

    Args:
        frame: Source image frame.
        bbox: Clamped bounding box (x_min, y_min, x_max, y_max).

    Returns:
        Cropped NumPy array.
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Extracted crop is empty for bbox: {bbox}")
    return crop


def normalize_pixels(
    image: np.ndarray,
    standardize: bool = False,
    mean: float = 0.421,
    std: float = 0.165,
) -> np.ndarray:
    """Normalizes pixel values into a deterministic numeric range.

    Default range: [0.0, 1.0] as float32.
    If standardize is True, applies z-score normalization: (x - mean) / std.

    Args:
        image: Grayscale or color image.
        standardize: Whether to apply dataset-level mean/std standardization.
        mean: Normalization mean (default: 0.421, standard for VSR lip crops).
        std: Normalization standard deviation (default: 0.165).

    Returns:
        Normalized float32 NumPy array.
    """
    img_f = image.astype(np.float32) / 255.0
    if standardize:
        return (img_f - mean) / (std + 1e-7)
    return img_f


class LipROIExtractor:
    """Stateful, deterministic mouth ROI extractor with landmark fallback tracking."""

    def __init__(
        self,
        image_size: int = 88,
        grayscale: bool = True,
        normalization: bool = True,
        padding_ratio: float = 0.25,
        square: bool = True,
    ):
        """Initializes the LipROIExtractor.

        Args:
            image_size: Spatial dimensions (image_size x image_size) for output ROI.
            grayscale: Whether to convert the crop to single-channel grayscale.
            normalization: Whether to normalize pixels to [0.0, 1.0].
            padding_ratio: Relative margin around detected mouth coordinates.
            square: Enforces square aspect ratio to prevent distortion.
        """
        self.image_size = image_size
        self.grayscale = grayscale
        self.normalization = normalization
        self.padding_ratio = padding_ratio
        self.square = square

        # Last successfully detected bounding box for temporal continuity fallback
        self._last_valid_bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self) -> None:
        """Resets temporal tracking state between distinct video sequences."""
        self._last_valid_bbox = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        """Extracts, crops, converts, resizes, and normalizes the lip ROI for one frame.

        Args:
            frame_bgr: Full-resolution OpenCV BGR frame.
            landmarks: Optional (N, 2) mouth landmark coordinates.

        Returns:
            Tuple of:
              - Processed 2D or 3D NumPy array of shape (image_size, image_size) or (image_size, image_size, C).
              - Boolean indicating whether fresh landmarks were detected (True) or fallback was used (False).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            blank = np.zeros((self.image_size, self.image_size), dtype=np.float32 if self.normalization else np.uint8)
            return blank, False

        h, w = frame_bgr.shape[:2]

        # Determine bounding box
        fresh_detection = False
        if landmarks is not None and len(landmarks) > 0:
            try:
                bbox = compute_mouth_bounding_box(
                    landmarks=landmarks,
                    frame_width=w,
                    frame_height=h,
                    padding_ratio=self.padding_ratio,
                    square=self.square,
                )
                self._last_valid_bbox = bbox
                fresh_detection = True
            except Exception as exc:
                logger.debug("Failed computing bbox from landmarks: %s. Attempting fallback.", exc)
                bbox = self._last_valid_bbox
        else:
            bbox = self._last_valid_bbox

        # If no valid previous detection exists, fallback to center-lower frame crop (mouth region heuristic)
        if bbox is None:
            crop_size = min(w, h) // 3
            center_x = w // 2
            center_y = int(h * 0.65)  # Human mouth typically at ~65% height
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            bbox = (x1, y1, x2, y2)
            logger.debug("No landmarks or history; using geometric center-mouth heuristic: %s", bbox)

        # Extract crop
        crop = extract_crop(frame_bgr, bbox)

        # Grayscale conversion
        if self.grayscale and len(crop.shape) == 3 and crop.shape[2] == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Resize to fixed image_size x image_size
        interp = cv2.INTER_AREA if crop.shape[0] >= self.image_size else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (self.image_size, self.image_size), interpolation=interp)

        # Pixel normalization
        if self.normalization:
            output = normalize_pixels(resized, standardize=False)
        else:
            output = resized

        return output, fresh_detection
