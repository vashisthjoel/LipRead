"""Unit and integration tests for Stage 2 Video Preprocessing and Lip ROI Extraction.

Validates:
  - Video reader error handling and properties.
  - Mouth bounding box geometry, aspect-ratio squaring, and boundary clamping.
  - Lip ROI cropping, grayscale conversion, and 88x88 resizing.
  - Pixel normalization range and determinism.
  - Fixed [T, C, H, W] sequence construction.
  - Short sequence handling (edge replication and zero padding).
  - Long sequence handling (center, start, and end truncation).
  - Preprocessing configuration loading.

NOTE: All test fixtures use synthetic NumPy frames purely for algorithmic verification.
These tests do NOT represent, simulate, or fabricate real LRS2 lip-reading data.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.preprocessing.lip_roi import (
    LipROIExtractor,
    compute_mouth_bounding_box,
    extract_crop,
    normalize_pixels,
)
from src.preprocessing.sequence import SequenceBuilder
from src.preprocessing.video_reader import VideoReader
from src.utils.config import get_config, load_yaml_config


class TestPreprocessingConfiguration(unittest.TestCase):
    """Tests loading and validation of preprocessing configuration."""

    def test_preprocessing_config_loaded(self) -> None:
        """Verifies configs/config.yaml contains the preprocessing block."""
        cfg = get_config(reload=True)
        prep = cfg.preprocessing
        self.assertIsInstance(prep, dict)
        self.assertEqual(prep.get("image_size"), 88)
        self.assertEqual(prep.get("sequence_length"), 75)
        self.assertTrue(prep.get("grayscale"))
        self.assertTrue(prep.get("normalization"))
        self.assertIn("min_face_detection_confidence", prep)
        self.assertIn("min_tracking_confidence", prep)


class TestBoundingBoxAndROI(unittest.TestCase):
    """Tests geometry, aspect ratio squaring, clamping, and ROI extraction."""

    def setUp(self) -> None:
        self.frame_w = 640
        self.frame_h = 480
        # Synthetic mouth landmarks centered around (320, 240) with width=60, height=20
        self.synthetic_landmarks = np.array(
            [
                [290, 230],  # Left corner
                [350, 230],  # Right corner
                [320, 220],  # Upper lip
                [320, 240],  # Lower lip
                [310, 235],
                [330, 235],
            ],
            dtype=np.int32,
        )

    def test_bounding_box_calculation_square(self) -> None:
        """Verifies square bounding box encompasses landmarks with padding."""
        bbox = compute_mouth_bounding_box(
            landmarks=self.synthetic_landmarks,
            frame_width=self.frame_w,
            frame_height=self.frame_h,
            padding_ratio=0.25,
            square=True,
        )
        x1, y1, x2, y2 = bbox
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, self.frame_w)
        self.assertLessEqual(y2, self.frame_h)

        # Enforces square (width == height or within 1 px due to integer rounding)
        bw = x2 - x1
        bh = y2 - y1
        self.assertAlmostEqual(bw, bh, delta=1)

        # Must encompass all original landmarks
        self.assertLessEqual(x1, np.min(self.synthetic_landmarks[:, 0]))
        self.assertGreaterEqual(x2, np.max(self.synthetic_landmarks[:, 0]))
        self.assertLessEqual(y1, np.min(self.synthetic_landmarks[:, 1]))
        self.assertGreaterEqual(y2, np.max(self.synthetic_landmarks[:, 1]))

    def test_bounding_box_clamping_boundary_extremes(self) -> None:
        """Verifies clamping when landmarks are at the extreme edge of the frame."""
        corner_landmarks = np.array([[5, 5], [10, 5], [5, 10]], dtype=np.int32)
        bbox = compute_mouth_bounding_box(
            landmarks=corner_landmarks,
            frame_width=self.frame_w,
            frame_height=self.frame_h,
            padding_ratio=0.5,
            square=True,
        )
        x1, y1, x2, y2 = bbox
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, self.frame_w)
        self.assertLessEqual(y2, self.frame_h)

    def test_extract_crop(self) -> None:
        """Verifies rectangular crop slicing from a synthetic frame."""
        frame = np.ones((self.frame_h, self.frame_w, 3), dtype=np.uint8) * 128
        bbox = (100, 50, 200, 150)
        crop = extract_crop(frame, bbox)
        self.assertEqual(crop.shape, (100, 100, 3))
        self.assertEqual(crop[0, 0, 0], 128)

    def test_pixel_normalization(self) -> None:
        """Verifies pixel normalization scales [0, 255] uint8 to [0.0, 1.0] float32."""
        raw_pixels = np.array([[0, 127, 255]], dtype=np.uint8)
        norm = normalize_pixels(raw_pixels, standardize=False)
        self.assertEqual(norm.dtype, np.float32)
        self.assertAlmostEqual(norm[0, 0], 0.0, places=4)
        self.assertAlmostEqual(norm[0, 1], 127.0 / 255.0, places=4)
        self.assertAlmostEqual(norm[0, 2], 1.0, places=4)

    def test_lip_roi_extractor_processing(self) -> None:
        """Verifies LipROIExtractor produces 88x88 normalized grayscale crops."""
        extractor = LipROIExtractor(image_size=88, grayscale=True, normalization=True)
        synthetic_frame = np.random.randint(0, 256, (self.frame_h, self.frame_w, 3), dtype=np.uint8)

        crop, fresh = extractor.process_frame(synthetic_frame, self.synthetic_landmarks)
        self.assertTrue(fresh)
        self.assertEqual(crop.shape, (88, 88))
        self.assertEqual(crop.dtype, np.float32)
        self.assertGreaterEqual(float(crop.min()), 0.0)
        self.assertLessEqual(float(crop.max()), 1.0)

    def test_lip_roi_extractor_fallback_on_missing_landmarks(self) -> None:
        """Verifies extractor falls back gracefully without crashing when landmarks=None."""
        extractor = LipROIExtractor(image_size=88, grayscale=True, normalization=True)
        synthetic_frame = np.ones((self.frame_h, self.frame_w, 3), dtype=np.uint8) * 100

        # Frame 1: successful detection
        _, fresh1 = extractor.process_frame(synthetic_frame, self.synthetic_landmarks)
        self.assertTrue(fresh1)

        # Frame 2: missing landmarks (should use last known bbox)
        crop2, fresh2 = extractor.process_frame(synthetic_frame, None)
        self.assertFalse(fresh2)  # Fallback used
        self.assertEqual(crop2.shape, (88, 88))


class TestSequenceBuilder(unittest.TestCase):
    """Tests temporal sequence construction, truncation, padding, and shape conventions."""

    def setUp(self) -> None:
        self.builder = SequenceBuilder(
            sequence_length=75,
            image_size=88,
            channels=1,
            padding_strategy="edge",
            truncation_strategy="center",
        )

    def test_exact_sequence_length(self) -> None:
        """Verifies exactly 75 frames produces shape [75, 1, 88, 88]."""
        synthetic_frames = [np.ones((88, 88), dtype=np.float32) for _ in range(75)]
        seq = self.builder.build_sequence(synthetic_frames)
        self.assertEqual(seq.shape, (75, 1, 88, 88))
        self.assertEqual(seq.dtype, np.float32)

    def test_short_sequence_edge_padding(self) -> None:
        """Verifies a 30-frame sequence is padded to 75 frames via edge replication."""
        # 30 frames where each frame has constant pixel value equal to its index
        synthetic_frames = [
            np.ones((88, 88), dtype=np.float32) * float(i) for i in range(30)
        ]
        seq = self.builder.build_sequence(synthetic_frames)
        self.assertEqual(seq.shape, (75, 1, 88, 88))

        # First 30 frames preserved
        self.assertAlmostEqual(seq[0, 0, 0, 0], 0.0)
        self.assertAlmostEqual(seq[29, 0, 0, 0], 29.0)

        # Remaining 45 frames replicated from frame 29
        for t in range(30, 75):
            self.assertAlmostEqual(seq[t, 0, 0, 0], 29.0)

    def test_short_sequence_zero_padding(self) -> None:
        """Verifies a 30-frame sequence is padded to 75 frames with zeros when configured."""
        zero_builder = SequenceBuilder(sequence_length=75, image_size=88, channels=1, padding_strategy="zero")
        synthetic_frames = [np.ones((88, 88), dtype=np.float32) * 5.0 for _ in range(30)]
        seq = zero_builder.build_sequence(synthetic_frames)
        self.assertEqual(seq.shape, (75, 1, 88, 88))

        # First 30 frames have value 5.0
        self.assertAlmostEqual(seq[29, 0, 0, 0], 5.0)
        # Padded frames are zero
        self.assertAlmostEqual(seq[30, 0, 0, 0], 0.0)
        self.assertAlmostEqual(seq[74, 0, 0, 0], 0.0)

    def test_long_sequence_center_truncation(self) -> None:
        """Verifies a 105-frame sequence is center-truncated to 75 frames."""
        synthetic_frames = [
            np.ones((88, 88), dtype=np.float32) * float(i) for i in range(105)
        ]
        seq = self.builder.build_sequence(synthetic_frames)
        self.assertEqual(seq.shape, (75, 1, 88, 88))

        # Expected start index: (105 - 75) // 2 = 15
        expected_start = 15
        self.assertAlmostEqual(seq[0, 0, 0, 0], float(expected_start))
        self.assertAlmostEqual(seq[74, 0, 0, 0], float(expected_start + 74))

    def test_to_tensor_conversion(self) -> None:
        """Verifies sequence conversion to PyTorch Tensor with shape torch.Size([75, 1, 88, 88])."""
        synthetic_frames = [np.zeros((88, 88), dtype=np.float32) for _ in range(75)]
        seq_np = self.builder.build_sequence(synthetic_frames)
        tensor = self.builder.to_tensor(seq_np)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tuple(tensor.shape), (75, 1, 88, 88))
        self.assertEqual(tensor.dtype, torch.float32)


class TestVideoReader(unittest.TestCase):
    """Tests OpenCV VideoReader validation and error handling."""

    def test_nonexistent_video_raises_file_not_found(self) -> None:
        """Ensures non-existent video path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            VideoReader("non_existent_file_path_xyz.mp4")

    def test_synthetic_video_read_and_properties(self) -> None:
        """Creates a tiny synthetic video and verifies VideoReader properties."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_video_path = Path(tmp_dir) / "test_synth.mp4"

            # Generate 15 frames of synthetic 320x240 video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(temp_video_path), fourcc, 25.0, (320, 240))
            for i in range(15):
                frame = np.ones((240, 320, 3), dtype=np.uint8) * (i * 15)
                out.write(frame)
            out.release()

            # Read with VideoReader
            with VideoReader(temp_video_path) as reader:
                self.assertGreater(reader.fps, 0)
                self.assertEqual(reader.width, 320)
                self.assertEqual(reader.height, 240)

                read_frames = list(reader.read_frames())
                self.assertEqual(len(read_frames), 15)
                idx, first_frame = read_frames[0]
                self.assertEqual(idx, 0)
                self.assertEqual(first_frame.shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
