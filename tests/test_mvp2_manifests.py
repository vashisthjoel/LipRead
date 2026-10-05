"""Unit and integration tests for MVP-2 manifests in LipRead."""

import json
from pathlib import Path
import unittest

METADATA_DIR = Path("E:/LipRead_Data/metadata")
TRAIN_PATH = METADATA_DIR / "mvp2_train.jsonl"
VAL_PATH = METADATA_DIR / "mvp2_val.jsonl"
TEST_PATH = METADATA_DIR / "mvp2_test.jsonl"

EXPECTED_TRAIN_SPKS = {
    "s1", "s2", "s10", "s11", "s12", "s13", "s15", "s16",
    "s17", "s18", "s22", "s24", "s25", "s26", "s27", "s28"
}
EXPECTED_VAL_SPKS = {"s19", "s20"}
EXPECTED_TEST_SPKS = {"s14", "s23"}


class TestMVP2Manifests(unittest.TestCase):
    """Test suite validating MVP-2 dataset manifests."""

    def test_manifest_existence_and_counts(self):
        """Verifies all 3 manifests exist and have exact approved sample counts."""
        for p, expected_count in [(TRAIN_PATH, 15000), (VAL_PATH, 2000), (TEST_PATH, 2000)]:
            self.assertTrue(p.exists(), f"Manifest {p} does not exist")
            with open(p, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), expected_count, f"Count mismatch for {p.name}")

    def test_speaker_partitions_disjoint_and_complete(self):
        """Verifies speaker sets match approved assignments and have zero leakage."""
        def get_speakers(p: Path):
            spks = set()
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    spks.add(json.loads(line)["speaker_id"])
            return spks

        train_spks = get_speakers(TRAIN_PATH)
        val_spks = get_speakers(VAL_PATH)
        test_spks = get_speakers(TEST_PATH)

        self.assertEqual(train_spks, EXPECTED_TRAIN_SPKS)
        self.assertEqual(val_spks, EXPECTED_VAL_SPKS)
        self.assertEqual(test_spks, EXPECTED_TEST_SPKS)

        self.assertTrue(train_spks.isdisjoint(val_spks))
        self.assertTrue(train_spks.isdisjoint(test_spks))
        self.assertTrue(val_spks.isdisjoint(test_spks))

    def test_no_duplicate_sample_ids(self):
        """Verifies no duplicate utterance IDs exist within any manifest."""
        for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
            ids = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    ids.append(json.loads(line)["sample_id"])
            self.assertEqual(len(ids), len(set(ids)), f"Duplicate IDs found in {p.name}")

    def test_record_structure_and_frame_counts(self):
        """Verifies frame counts, frame paths length, and non-empty transcripts."""
        for p, split_name in [(TRAIN_PATH, "train"), (VAL_PATH, "val"), (TEST_PATH, "test")]:
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    rec = json.loads(line)
                    self.assertEqual(rec["split"], split_name)
                    self.assertEqual(rec["frame_count"], 75)
                    self.assertEqual(len(rec["frame_paths"]), 75)
                    self.assertTrue(bool(rec["transcript"].strip()))
                    if i >= 50:  # Spot-check first 50 per file to keep test fast
                        break


if __name__ == "__main__":
    unittest.main()
