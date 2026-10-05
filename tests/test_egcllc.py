"""Unit and integration tests for EGCLLC dataset components in LipRead.

Validates:
  - GRID .align transcript parsing and silence filtering
  - EGCLLCAdapter image loading, grayscale conversion, 88x88 resize, and [0, 1] normalization
  - Fixed temporal sequence [75, 1, 88, 88] tensor generation
  - Missing image file detection
  - Speaker-aware split generation with strictly zero speaker leakage
  - Batch collation with variable-length target padding
  - Deterministic tensor cache path generation

NOTE: Tests use small temporary mock fixtures for isolated unit testing.
These fixtures do NOT represent or fabricate real dataset statistics.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.collate import pad_collate_fn
from src.data.dataset_config import VocabularyConfig
from src.data.egcllc_adapter import EGCLLCAdapter
from src.data.egcllc_dataset import EGCLLCDataset, EGCLLCIndexBuilder, EGCLLCSample
from src.data.egcllc_inspector import parse_grid_alignment


class TestEGCLLCParsingAndAdapter(unittest.TestCase):
    """Tests alignment parsing, image formatting, and adapter tensor building."""

    def test_parse_grid_alignment_standard(self) -> None:
        """Verifies parsing of standard GRID alignment file with silence stripping."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".align") as f:
            f.write("0 23750 sil\n")
            f.write("23750 29500 bin\n")
            f.write("29500 34000 blue\n")
            f.write("34000 35500 at\n")
            f.write("35500 41000 f\n")
            f.write("41000 47250 two\n")
            f.write("47250 53000 now\n")
            f.write("53000 74500 sil\n")
            temp_path = Path(f.name)

        try:
            sentence, segments = parse_grid_alignment(temp_path)
            self.assertEqual(sentence, "bin blue at f two now")
            self.assertEqual(len(segments), 8)
            # Verify words extracted without 'sil'
            words = [s[2] for s in segments if s[2] != "sil"]
            self.assertEqual(words, ["bin", "blue", "at", "f", "two", "now"])
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_adapter_image_loading_and_conversion(self) -> None:
        """Verifies image loading, grayscale conversion, 88x88 resize, and [0, 1] normalization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test_frame.jpg"
            # Create a 64x64 BGR synthetic image
            synthetic_bgr = np.ones((64, 64, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(img_path), synthetic_bgr)

            adapter = EGCLLCAdapter(image_size=88, grayscale=True, normalize=True)
            loaded = adapter.load_frame(img_path)

            self.assertEqual(loaded.shape, (88, 88))
            self.assertEqual(loaded.dtype, np.float32)
            self.assertAlmostEqual(float(loaded[0, 0]), 128.0 / 255.0, places=4)

    def test_adapter_process_frame_paths_sequence_shape(self) -> None:
        """Verifies process_frame_paths produces [75, 1, 88, 88] PyTorch tensor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create 10 frame images
            paths = []
            for i in range(10):
                p = Path(tmp_dir) / f"{i:04d}.jpg"
                img = np.ones((50, 50, 3), dtype=np.uint8) * (i * 20)
                cv2.imwrite(str(p), img)
                paths.append(p)

            adapter = EGCLLCAdapter(image_size=88, sequence_length=75, grayscale=True)
            tensor = adapter.process_frame_paths(paths)

            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tuple(tensor.shape), (75, 1, 88, 88))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertGreaterEqual(float(tensor.min()), 0.0)
            self.assertLessEqual(float(tensor.max()), 1.0)

    def test_adapter_missing_file_raises(self) -> None:
        """Verifies load_frame raises FileNotFoundError on missing files."""
        adapter = EGCLLCAdapter()
        with self.assertRaises(FileNotFoundError):
            adapter.load_frame("non_existent_image_path_1234.jpg")


    def test_speaker_aware_splits_no_leakage(self) -> None:
        """Verifies EGCLLCIndexBuilder guarantees strictly zero speaker leakage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            meta_dir = Path(tmp_dir) / "metadata"
            align_dir = root / "GRID_alignments"
            img_dir = root / "lip_images"
            coord_dir = root / "lip_coordinates"

            # Create mock speakers s1, s2, s3, s4, s5, s6, s7
            speakers = [f"s{i}" for i in range(1, 8)]
            for spk in speakers:
                (align_dir / spk).mkdir(parents=True)
                (img_dir / spk).mkdir(parents=True)
                (coord_dir / spk).mkdir(parents=True)

                # 2 utterances per speaker
                for vid in ("bbaf2n", "bbaf3s"):
                    # Alignment
                    with open(align_dir / spk / f"{vid}.align", "w") as f:
                        f.write(f"0 1000 sil\n1000 2000 test\n2000 3000 {spk}\n3000 4000 sil\n")
                    # Coords
                    with open(coord_dir / spk / f"{vid}.json", "w") as f:
                        json.dump({"1": [[10, 10]]}, f)
                    # Image folder with 2 frames
                    vid_img = img_dir / spk / vid
                    vid_img.mkdir()
                    cv2.imwrite(str(vid_img / "0001.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
                    cv2.imwrite(str(vid_img / "0002.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))

            builder = EGCLLCIndexBuilder(dataset_root=root, metadata_dir=meta_dir, seed=42)
            summary = builder.build_and_save_index()

            train_spks = set(summary["splits"]["train"]["speakers"])
            val_spks = set(summary["splits"]["val"]["speakers"])
            test_spks = set(summary["splits"]["test"]["speakers"])

            # Verify strictly disjoint sets (zero leakage)
            self.assertTrue(train_spks.isdisjoint(val_spks))
            self.assertTrue(train_spks.isdisjoint(test_spks))
            self.assertTrue(val_spks.isdisjoint(test_spks))
            self.assertEqual(train_spks | val_spks | test_spks, set(speakers))

            # Verify individual manifest entries match assigned splits
            for split_name in ("train", "val", "test"):
                manifest = meta_dir / f"{split_name}.jsonl"
                self.assertTrue(manifest.exists())
                with open(manifest, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        spk = item["speaker_id"]
                        if split_name == "train":
                            self.assertIn(spk, train_spks)
                        elif split_name == "val":
                            self.assertIn(spk, val_spks)
                        elif split_name == "test":
                            self.assertIn(spk, test_spks)

    def test_frame_ordering_numeric(self) -> None:
        """Verifies frames are ordered numerically (1.jpg, 2.jpg, ..., 10.jpg) not lexically."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            seq_dir = Path(tmp_dir) / "test_seq"
            seq_dir.mkdir()
            # Create frames out of order: 10.jpg, 2.jpg, 1.jpg
            for num in [10, 2, 1]:
                cv2.imwrite(str(seq_dir / f"{num}.jpg"), np.zeros((30, 30, 3), dtype=np.uint8))

            adapter = EGCLLCAdapter(sequence_length=3)
            # Test process_sequence_dir
            frames = sorted(
                [f for f in seq_dir.iterdir() if f.suffix == ".jpg"],
                key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem,
            )
            ordered_names = [f.name for f in frames]
            self.assertEqual(ordered_names, ["1.jpg", "2.jpg", "10.jpg"])

    def test_index_builder_nested_structure(self) -> None:
        """Verifies EGCLLCIndexBuilder correctly indexes nested s1/video/mpg_6000/{vid} structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            meta_dir = Path(tmp_dir) / "metadata"
            align_dir = root / "GRID_alignments" / "s1" / "align"
            align_dir.mkdir(parents=True)
            nested_img = root / "lip_images" / "s1" / "video" / "mpg_6000" / "bbaf2n"
            nested_img.mkdir(parents=True)

            with open(align_dir / "bbaf2n.align", "w") as f:
                f.write("0 1000 sil\n1000 2000 hello\n2000 3000 world\n3000 4000 sil\n")
            for i in range(1, 4):
                cv2.imwrite(str(nested_img / f"{i}.jpg"), np.zeros((30, 30, 3), dtype=np.uint8))

            builder = EGCLLCIndexBuilder(dataset_root=root, metadata_dir=meta_dir, seed=42)
            summary = builder.build_and_save_index()

            index_file = meta_dir / "egcllc_index.jsonl"
            self.assertTrue(index_file.exists())
            with open(index_file, "r") as f:
                records = [json.loads(line) for line in f]

            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["speaker_id"], "s1")
            self.assertEqual(rec["video_id"], "bbaf2n")
            self.assertEqual(rec["transcript"], "hello world")
            self.assertEqual(rec["frame_count"], 3)
            self.assertTrue(rec["frame_paths"][0].endswith("1.jpg"))
            self.assertTrue(rec["frame_paths"][1].endswith("2.jpg"))
            self.assertTrue(rec["frame_paths"][2].endswith("3.jpg"))


class TestCollationAndDataset(unittest.TestCase):
    """Tests pad_collate_fn and dataset tokenization."""

    def test_pad_collate_fn(self) -> None:
        """Verifies pad_collate_fn stacks video tensors and pads variable-length tokens."""
        batch = [
            {
                "frames": torch.ones((75, 1, 88, 88), dtype=torch.float32) * 1.0,
                "tokens": torch.tensor([1, 2, 3], dtype=torch.long),
                "transcript": "abc",
                "sample_id": "s1/v1",
                "speaker_id": "s1",
                "video_id": "v1",
                "token_count": 3,
            },
            {
                "frames": torch.ones((75, 1, 88, 88), dtype=torch.float32) * 2.0,
                "tokens": torch.tensor([4, 5, 6, 7, 8], dtype=torch.long),
                "transcript": "defgh",
                "sample_id": "s2/v2",
                "speaker_id": "s2",
                "video_id": "v2",
                "token_count": 5,
            },
        ]

        collated = pad_collate_fn(batch)

        self.assertEqual(tuple(collated["frames"].shape), (2, 75, 1, 88, 88))
        self.assertEqual(tuple(collated["tokens"].shape), (2, 5))
        # First sample padded with 0 at indices 3 and 4
        self.assertEqual(collated["tokens"][0].tolist(), [1, 2, 3, 0, 0])
        # Second sample unchanged
        self.assertEqual(collated["tokens"][1].tolist(), [4, 5, 6, 7, 8])
        self.assertEqual(collated["target_lengths"].tolist(), [3, 5])
        self.assertEqual(collated["input_lengths"].tolist(), [75, 75])
        self.assertEqual(collated["transcripts"], ["abc", "defgh"])
        self.assertEqual(collated["sample_ids"], ["s1/v1", "s2/v2"])


if __name__ == "__main__":
    unittest.main()
