"""Sequential OpenCV video frame reader for LipRead.

Provides memory-efficient streaming frame extraction, validation, and metadata
introspection using OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger("lipread.preprocessing.video_reader")


class VideoReader:
    """Memory-efficient, sequential OpenCV video reader.

    Streams frames without buffering entire video files into memory,
    with comprehensive error handling and metadata inspection.
    """

    def __init__(self, video_path: Union[str, Path]):
        """Initializes and validates the video reader.

        Args:
            video_path: Absolute or relative path to the input video.

        Raises:
            FileNotFoundError: If the specified video path does not exist.
            ValueError: If OpenCV cannot open the video file.
        """
        self.video_path = Path(video_path).resolve()
        if not self.video_path.exists():
            logger.error("Video file does not exist: %s", self.video_path)
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        if not self.video_path.is_file():
            logger.error("Path is not a regular file: %s", self.video_path)
            raise ValueError(f"Path is not a regular file: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            logger.error("OpenCV failed to open video file: %s", self.video_path)
            raise ValueError(
                f"Failed to open video file (corrupted or unsupported codec): {self.video_path}"
            )

        # Inspect video properties
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps: float = float(raw_fps) if raw_fps > 0 else 25.0
        if raw_fps <= 0:
            logger.warning(
                "Video reported invalid FPS (%s). Defaulting to 25.0 fps for %s",
                raw_fps,
                self.video_path.name,
            )

        raw_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._frame_count: int = max(0, int(raw_count)) if raw_count > 0 else 0

        self._width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.debug(
            "Opened video: %s | FPS: %.2f | Frames: %d | Resolution: %dx%d",
            self.video_path.name,
            self._fps,
            self._frame_count,
            self._width,
            self._height,
        )

    @property
    def fps(self) -> float:
        """Video framerate (frames per second)."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total frame count reported by video container (0 if unavailable)."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def duration_seconds(self) -> float:
        """Estimated duration in seconds based on header metadata."""
        if self._fps > 0 and self._frame_count > 0:
            return self._frame_count / self._fps
        return 0.0

    def read_frames(
        self, max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iteratively yields frames one by one in temporal order.

        Does not buffer the full video in memory.

        Args:
            max_frames: Optional upper limit on number of frames to yield.

        Yields:
            Tuple of (frame_index, bgr_frame_numpy_array).
        """
        if not self._cap.isOpened():
            logger.error("Cannot read frames: VideoCapture is closed.")
            return

        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                logger.debug("Reached max_frames limit (%d)", max_frames)
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.debug(
                    "End of video stream or unreadable frame encountered at index %d",
                    frame_idx,
                )
                break

            yield frame_idx, frame
            frame_idx += 1

    def read_all_frames(self, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Convenience helper to read all frames into a list.

        Args:
            max_frames: Optional maximum frame limit.

        Returns:
            List of BGR frames as NumPy arrays.
        """
        return [frame for _, frame in self.read_frames(max_frames=max_frames)]

    def seek(self, frame_idx: int) -> bool:
        """Seeks the video stream to the specified zero-indexed frame.

        Args:
            frame_idx: Target frame index.

        Returns:
            True if seek command succeeded, False otherwise.
        """
        if not self._cap.isOpened():
            return False
        return bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx)))

    def close(self) -> None:
        """Releases the underlying OpenCV VideoCapture resource."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.debug("Released VideoCapture for %s", self.video_path.name)

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"VideoReader(path='{self.video_path.name}', fps={self._fps:.2f}, "
            f"frames={self._frame_count}, res={self._width}x{self._height})"
        )
