"""Basic environment and foundation tests for LipRead.

Validates:
  - Python runtime version
  - PyTorch installation and tensor operations
  - CUDA availability logic
  - Central configuration loading from YAML
  - Data path resolution and defaults
  - Vocabulary tokenization structures
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from src.data.dataset_config import VocabularyConfig
from src.data.paths import (
    get_checkpoint_dir,
    get_data_root,
    get_metadata_dir,
    get_processed_data_dir,
    get_project_root,
    get_raw_data_dir,
)
from src.utils.config import Config, get_config, load_yaml_config


class TestEnvironmentAndFoundation(unittest.TestCase):
    """Test suite for foundation integrity."""

    def test_python_version(self) -> None:
        """Ensures Python version is at least 3.10."""
        v = sys.version_info
        self.assertGreaterEqual(
            (v.major, v.minor),
            (3, 10),
            f"Python >= 3.10 is required, but found {v.major}.{v.minor}.{v.micro}",
        )

    def test_pytorch_import_and_operations(self) -> None:
        """Ensures PyTorch is importable and can perform basic tensor math."""
        import torch

        self.assertTrue(hasattr(torch, "__version__"))
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        self.assertEqual(z.tolist(), [5.0, 7.0, 9.0])

    def test_cuda_detection_logic(self) -> None:
        """Verifies CUDA detection API and device query."""
        import torch

        is_cuda = torch.cuda.is_available()
        self.assertIsInstance(is_cuda, bool)

        if is_cuda:
            device_count = torch.cuda.device_count()
            self.assertGreater(device_count, 0)
            device_name = torch.cuda.get_device_name(0)
            self.assertIsInstance(device_name, str)
            self.assertGreater(len(device_name), 0)

            # Test basic CUDA tensor allocation
            cuda_tensor = torch.ones((2, 2), device="cuda")
            self.assertEqual(cuda_tensor.device.type, "cuda")

    def test_configuration_loading(self) -> None:
        """Verifies configs/config.yaml loads properly with required sections."""
        cfg_dict = load_yaml_config()
        self.assertIsInstance(cfg_dict, dict)
        self.assertIn("project", cfg_dict)
        self.assertIn("data", cfg_dict)
        self.assertIn("video", cfg_dict)
        self.assertIn("model", cfg_dict)
        self.assertIn("training", cfg_dict)

        cfg = get_config(reload=True)
        self.assertEqual(cfg.project.get("name"), "LipRead")
        self.assertEqual(cfg.video.get("image_size"), [88, 88])
        self.assertIn(cfg.training.get("batch_size"), (2, 4))

    def test_data_paths_resolution(self) -> None:
        """Verifies path resolution returns valid Path instances."""
        proj_root = get_project_root()
        self.assertTrue(proj_root.exists())
        self.assertTrue((proj_root / "src").exists())

        data_root = get_data_root()
        self.assertIsInstance(data_root, Path)

        raw_dir = get_raw_data_dir()
        self.assertIsInstance(raw_dir, Path)
        self.assertEqual(raw_dir.name, "raw")

        proc_dir = get_processed_data_dir()
        self.assertIsInstance(proc_dir, Path)
        self.assertEqual(proc_dir.name, "processed")

        meta_dir = get_metadata_dir()
        self.assertIsInstance(meta_dir, Path)
        self.assertEqual(meta_dir.name, "metadata")

        ckpt_dir = get_checkpoint_dir()
        self.assertIsInstance(ckpt_dir, Path)
        self.assertEqual(ckpt_dir.name, "checkpoints")

    def test_vocabulary_encoding_decoding(self) -> None:
        """Verifies vocabulary encoding/decoding and CTC blank token handling."""
        vocab = VocabularyConfig()
        test_phrase = "hello world"
        encoded = vocab.encode(test_phrase)

        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), len(test_phrase))
        # Blank index is 0, so valid character tokens must be > 0
        for token_id in encoded:
            self.assertGreater(token_id, 0)

        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, test_phrase)

    def test_optional_cv_packages_not_breaking(self) -> None:
        """Ensures missing OpenCV / MediaPipe does not raise unexpected errors."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pass  # Expected if not yet installed in foundation stage

        try:
            import mediapipe  # noqa: F401
        except ImportError:
            pass  # Expected if not yet installed in foundation stage


if __name__ == "__main__":
    unittest.main()
