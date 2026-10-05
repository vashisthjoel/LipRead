"""Facial landmark and mouth/lip localization using MediaPipe for LipRead.

Provides modular landmark extraction, converting raw video frames into
normalized pixel coordinates for lip region-of-interest (ROI) tracking.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT
from src.utils.logging_config import get_logger

logger = get_logger("lipread.preprocessing.landmarks")

# Canonical MediaPipe Face Mesh landmark indices for outer and inner lips (40 landmarks)
MOUTH_LANDMARK_INDICES: List[int] = [
    # Outer lips
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Inner lips
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model_asset(model_path: Optional[Union[str, Path]] = None) -> Path:
    """Ensures the MediaPipe FaceLandmarker model task asset is available locally.

    Args:
        model_path: Optional target path. Defaults to PROJECT_ROOT / 'weights' / 'face_landmarker.task'.

    Returns:
        Path to verified model task file.
    """
    if model_path is not None:
        target = Path(model_path).resolve()
    else:
        target = (PROJECT_ROOT / "weights" / "face_landmarker.task").resolve()

    if not target.exists():
        logger.info("MediaPipe model not found at %s. Downloading official model asset...", target)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(OFFICIAL_MODEL_URL, str(target))
            logger.info("Successfully downloaded face_landmarker.task to %s", target)
        except Exception as exc:
            logger.error("Failed to download MediaPipe model: %s", exc)
            raise RuntimeError(
                f"MediaPipe model asset required but could not be downloaded to {target}: {exc}"
            ) from exc

    return target


class LandmarkProcessor:
    """Detects facial landmarks and isolates lip coordinates from OpenCV frames."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_asset_path: Optional[Union[str, Path]] = None,
    ):
        """Initializes the MediaPipe face landmark detector.

        Args:
            min_detection_confidence: Minimum confidence threshold for face detection.
            min_tracking_confidence: Minimum confidence threshold for landmark tracking.
            model_asset_path: Path to the face landmarker model task file.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = ensure_model_asset(model_asset_path)

        self._detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes MediaPipe FaceLandmarker using the Tasks API."""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._detector = FaceLandmarker.create_from_options(options)
            logger.debug("Initialized MediaPipe FaceLandmarker with %s", self.model_path.name)
        except Exception as exc:
            logger.error("Failed to initialize MediaPipe FaceLandmarker: %s", exc)
            raise

    def detect_lip_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detects mouth/lip landmarks in a single BGR OpenCV frame.

        Args:
            frame_bgr: Input image in OpenCV BGR format with shape (H, W, 3).

        Returns:
            NumPy array of shape (N, 2) containing (x, y) pixel coordinates of the lips,
            or None if no face was detected in the frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Empty or None frame passed to detect_lip_landmarks.")
            return None

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return None

        # MediaPipe expects RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self._detector.detect(mp_image)

            if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                logger.debug("No face detected in frame.")
                return None

            # Primary face landmarks
            face = detection_result.face_landmarks[0]

            # Extract lip landmark coordinates scaled to pixel dimensions
            lip_coords = []
            for idx in MOUTH_LANDMARK_INDICES:
                lm = face[idx]
                x_px = int(round(lm.x * w))
                y_px = int(round(lm.y * h))
                lip_coords.append([x_px, y_px])

            return np.array(lip_coords, dtype=np.int32)

        except Exception as exc:
            logger.warning("Error during landmark detection on frame: %s", exc)
            return None

    def detect_all_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detects all facial mesh landmarks in a single BGR OpenCV frame.

        Args:
            frame_bgr: Input image in BGR format.

        Returns:
            NumPy array of shape (468/478, 2) with (x, y) pixel coordinates,
            or None if no face detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self._detector.detect(mp_image)

            if not detection_result.face_landmarks:
                return None

            face = detection_result.face_landmarks[0]
            coords = [[int(round(lm.x * w)), int(round(lm.y * h))] for lm in face]
            return np.array(coords, dtype=np.int32)
        except Exception as exc:
            logger.warning("Error detecting all landmarks: %s", exc)
            return None

    def close(self) -> None:
        """Releases the underlying MediaPipe landmarker."""
        if self._detector is not None:
            try:
                self._detector.close()
                logger.debug("Closed MediaPipe FaceLandmarker.")
            except Exception:
                pass
            self._detector = None

    def __enter__(self) -> LandmarkProcessor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
