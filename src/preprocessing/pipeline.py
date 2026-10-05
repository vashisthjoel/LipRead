"""High-level end-to-end video preprocessing pipeline for LipRead.

Integrates:
    VideoReader -> LandmarkProcessor -> LipROIExtractor -> SequenceBuilder

Transforms arbitrary input video files into standardized [T, C, H, W] tensors
ready for 3D CNN + Bi-LSTM training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.preprocessing.landmarks import LandmarkProcessor
from src.preprocessing.lip_roi import LipROIExtractor
from src.preprocessing.sequence import SequenceBuilder
from src.preprocessing.video_reader import VideoReader
from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger("lipread.preprocessing.pipeline")


@dataclass
class PreprocessingResult:
    """Detailed summary and tensor output produced by LipPreprocessingPipeline."""

    source_video: Path
    original_fps: float
    original_frame_count: int
    frames_processed: int
    frames_with_face: int
    frames_with_lip: int
    output_shape: Optional[Tuple[int, ...]]
    success: bool
    tensor: Optional[torch.Tensor] = None
    error_message: Optional[str] = None

    @property
    def face_detection_rate(self) -> float:
        """Fraction of processed frames with successful face detection."""
        if self.frames_processed == 0:
            return 0.0
        return self.frames_with_face / self.frames_processed

    @property
    def lip_extraction_rate(self) -> float:
        """Fraction of processed frames with valid lip extraction."""
        if self.frames_processed == 0:
            return 0.0
        return self.frames_with_lip / self.frames_processed

    def summary(self) -> str:
        """Returns a formatted multi-line summary string."""
        status = "SUCCESS" if self.success else f"FAILED ({self.error_message})"
        shape_str = str(self.output_shape) if self.output_shape else "None"
        return (
            f"--- Preprocessing Result ---\n"
            f"Source Video:           {self.source_video.name}\n"
            f"Status:                 {status}\n"
            f"Original FPS:           {self.original_fps:.2f}\n"
            f"Original Frame Count:   {self.original_frame_count}\n"
            f"Frames Processed:       {self.frames_processed}\n"
            f"Face Detection Count:   {self.frames_with_face} ({self.face_detection_rate * 100:.1f}%)\n"
            f"Lip Extraction Count:   {self.frames_with_lip} ({self.lip_extraction_rate * 100:.1f}%)\n"
            f"Output Tensor Shape:    {shape_str} [T, C, H, W]\n"
            f"----------------------------"
        )


class LipPreprocessingPipeline:
    """Production-grade pipeline executing full video-to-tensor preprocessing."""

    def __init__(
        self,
        image_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        grayscale: Optional[bool] = None,
        normalization: Optional[bool] = None,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        padding_ratio: Optional[float] = None,
        padding_strategy: Optional[str] = None,
        truncation_strategy: Optional[str] = None,
        model_asset_path: Optional[Union[str, Path]] = None,
    ):
        """Initializes the preprocessing pipeline from central configuration and overrides.

        Args:
            image_size: Target spatial dimensions (default from config: 88).
            sequence_length: Target temporal frame count (default from config: 75).
            grayscale: Whether to convert ROI to grayscale (default: True).
            normalization: Whether to normalize pixels to [0.0, 1.0] (default: True).
            min_detection_confidence: Landmark face detection confidence threshold.
            min_tracking_confidence: Landmark tracking confidence threshold.
            padding_ratio: Relative margin around detected lips (default: 0.25).
            padding_strategy: 'edge' or 'zero' for sequences shorter than sequence_length.
            truncation_strategy: 'center', 'start', or 'end' for sequences longer than sequence_length.
            model_asset_path: Optional path to MediaPipe FaceLandmarker task asset.
        """
        cfg = get_config().preprocessing

        self.image_size = image_size if image_size is not None else cfg.get("image_size", 88)
        self.sequence_length = (
            sequence_length if sequence_length is not None else cfg.get("sequence_length", 75)
        )
        self.grayscale = grayscale if grayscale is not None else cfg.get("grayscale", True)
        self.normalization = (
            normalization if normalization is not None else cfg.get("normalization", True)
        )
        self.padding_ratio = (
            padding_ratio if padding_ratio is not None else cfg.get("padding_ratio", 0.25)
        )
        self.padding_strategy = (
            padding_strategy if padding_strategy is not None else cfg.get("padding_strategy", "edge")
        )
        self.truncation_strategy = (
            truncation_strategy
            if truncation_strategy is not None
            else cfg.get("truncation_strategy", "center")
        )

        det_conf = (
            min_detection_confidence
            if min_detection_confidence is not None
            else cfg.get("min_face_detection_confidence", 0.5)
        )
        track_conf = (
            min_tracking_confidence
            if min_tracking_confidence is not None
            else cfg.get("min_tracking_confidence", 0.5)
        )
        model_path = model_asset_path or cfg.get("model_asset_path", None)

        # Initialize modular stages
        self.landmark_processor = LandmarkProcessor(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            model_asset_path=model_path,
        )
        self.roi_extractor = LipROIExtractor(
            image_size=self.image_size,
            grayscale=self.grayscale,
            normalization=self.normalization,
            padding_ratio=self.padding_ratio,
            square=True,
        )
        self.sequence_builder = SequenceBuilder(
            sequence_length=self.sequence_length,
            image_size=self.image_size,
            channels=1 if self.grayscale else 3,
            padding_strategy=self.padding_strategy,
            truncation_strategy=self.truncation_strategy,
        )

        logger.debug(
            "Initialized LipPreprocessingPipeline (size=%dx%d, seq_len=%d, grayscale=%s)",
            self.image_size,
            self.image_size,
            self.sequence_length,
            self.grayscale,
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
    ) -> PreprocessingResult:
        """Processes an input video end-to-end into a normalized [T, C, H, W] tensor.

        Args:
            video_path: Path to the target video file.
            max_frames: Optional upper limit on number of input frames to read.

        Returns:
            PreprocessingResult containing tensor and detailed extraction statistics.
        """
        path = Path(video_path).resolve()
        if not path.exists():
            return PreprocessingResult(
                source_video=path,
                original_fps=0.0,
                original_frame_count=0,
                frames_processed=0,
                frames_with_face=0,
                frames_with_lip=0,
                output_shape=None,
                success=False,
                error_message=f"Video file not found: {path}",
            )

        try:
            with VideoReader(path) as reader:
                fps = reader.fps
                total_frames = reader.frame_count

                self.roi_extractor.reset()
                processed_roi_frames: List[np.ndarray] = []
                frames_processed = 0
                frames_with_face = 0
                frames_with_lip = 0

                for _, frame_bgr in reader.read_frames(max_frames=max_frames):
                    frames_processed += 1

                    # 1. Detect facial / lip landmarks
                    lip_landmarks = self.landmark_processor.detect_lip_landmarks(frame_bgr)
                    if lip_landmarks is not None and len(lip_landmarks) > 0:
                        frames_with_face += 1

                    # 2. Extract and normalize mouth ROI (with fallback tracking)
                    roi_frame, fresh_detection = self.roi_extractor.process_frame(
                        frame_bgr=frame_bgr,
                        landmarks=lip_landmarks,
                    )
                    if fresh_detection:
                        frames_with_lip += 1

                    processed_roi_frames.append(roi_frame)

                if frames_processed == 0:
                    return PreprocessingResult(
                        source_video=path,
                        original_fps=fps,
                        original_frame_count=total_frames,
                        frames_processed=0,
                        frames_with_face=0,
                        frames_with_lip=0,
                        output_shape=None,
                        success=False,
                        error_message="Video contains zero readable frames.",
                    )

                # 3. Assemble fixed-length sequence [T, C, H, W]
                seq_np = self.sequence_builder.build_sequence(processed_roi_frames)
                tensor = self.sequence_builder.to_tensor(seq_np)

                logger.info(
                    "Processed %s | Frames: %d | Faces: %d | Output: %s",
                    path.name,
                    frames_processed,
                    frames_with_face,
                    tuple(tensor.shape),
                )

                return PreprocessingResult(
                    source_video=path,
                    original_fps=fps,
                    original_frame_count=total_frames,
                    frames_processed=frames_processed,
                    frames_with_face=frames_with_face,
                    frames_with_lip=frames_with_lip,
                    output_shape=tuple(tensor.shape),
                    success=True,
                    tensor=tensor,
                )

        except Exception as exc:
            logger.error("Preprocessing failed for %s: %s", path, exc, exc_info=True)
            return PreprocessingResult(
                source_video=path,
                original_fps=0.0,
                original_frame_count=0,
                frames_processed=0,
                frames_with_face=0,
                frames_with_lip=0,
                output_shape=None,
                success=False,
                error_message=str(exc),
            )

    def close(self) -> None:
        """Releases underlying MediaPipe detector resources."""
        if self.landmark_processor is not None:
            self.landmark_processor.close()

    def __enter__(self) -> LipPreprocessingPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
