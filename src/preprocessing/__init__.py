"""Video preprocessing and lip region-of-interest (ROI) extraction package.

Exports:
    VideoReader: Sequential OpenCV video frame reader.
    LandmarkProcessor: MediaPipe facial landmarking and mouth localization.
    LipROIExtractor: Padded mouth bounding box, crop, resize (88x88), and normalization.
    SequenceBuilder: Standardized [T, C, H, W] fixed temporal tensor builder.
    LipPreprocessingPipeline: End-to-end video-to-tensor preprocessing pipeline.
    PreprocessingResult: Metadata and tensor container for pipeline runs.
"""

from src.preprocessing.landmarks import LandmarkProcessor
from src.preprocessing.lip_roi import (
    LipROIExtractor,
    compute_mouth_bounding_box,
    extract_crop,
    normalize_pixels,
)
from src.preprocessing.pipeline import LipPreprocessingPipeline, PreprocessingResult
from src.preprocessing.sequence import SequenceBuilder
from src.preprocessing.video_reader import VideoReader

__all__ = [
    "VideoReader",
    "LandmarkProcessor",
    "LipROIExtractor",
    "compute_mouth_bounding_box",
    "extract_crop",
    "normalize_pixels",
    "SequenceBuilder",
    "LipPreprocessingPipeline",
    "PreprocessingResult",
]
