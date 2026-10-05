"""Script to deterministically create MVP-2 manifests for LipRead.

Approved Split Specifications:
- Train: 16 speakers (s1, s2, s10, s11, s12, s13, s15, s16, s17, s18, s22, s24, s25, s26, s27, s28)
         Exactly 15,000 samples, sampled deterministically with seed 42.
- Validation: 2 speakers (s19, s20)
              Exactly 2,000 samples (1,000 from s19, 1,000 from s20).
- Test: 2 speakers (s14, s23)
        Exactly 2,000 samples (1,000 from s14, 1,000 from s23).
- Excluded: s29 and all missing speakers.

Outputs:
- E:\\LipRead_Data\\metadata\\mvp2_train.jsonl
- E:\\LipRead_Data\\metadata\\mvp2_val.jsonl
- E:\\LipRead_Data\\metadata\\mvp2_test.jsonl
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
METADATA_DIR = Path("E:/LipRead_Data/metadata")
RAW_IMG_ROOT = Path("E:/LipRead_Data/raw/EGCLLC/lip_images")
ALIGN_ROOT = Path("E:/LipRead_Data/raw/EGCLLC/GRID_alignments")
COORD_ROOT = Path("E:/LipRead_Data/raw/EGCLLC/lip_coordinates")
INDEX_PATH = METADATA_DIR / "egcllc_index.jsonl"

OUT_TRAIN = METADATA_DIR / "mvp2_train.jsonl"
OUT_VAL = METADATA_DIR / "mvp2_val.jsonl"
OUT_TEST = METADATA_DIR / "mvp2_test.jsonl"

TRAIN_SPEAKERS = [
    "s1", "s2", "s10", "s11", "s12", "s13", "s15", "s16",
    "s17", "s18", "s22", "s24", "s25", "s26", "s27", "s28"
]
VAL_SPEAKERS = ["s19", "s20"]
TEST_SPEAKERS = ["s14", "s23"]

TRAIN_SPKS_SET = set(TRAIN_SPEAKERS)
VAL_SPKS_SET = set(VAL_SPEAKERS)
TEST_SPKS_SET = set(TEST_SPEAKERS)
ALL_APPROVED_SET = TRAIN_SPKS_SET | VAL_SPKS_SET | TEST_SPKS_SET


def create_manifests() -> None:
    print("==================================================")
    print("LIPREAD — CREATING MVP-2 MANIFESTS")
    print("==================================================")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Master index not found at {INDEX_PATH}")

    # Read records from master index for approved speakers
    print(f"Reading master index from {INDEX_PATH}...")
    records_by_spk = {s: [] for s in ALL_APPROVED_SET}

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            spk = rec.get("speaker_id")
            if spk in ALL_APPROVED_SET:
                trans = rec.get("transcript", "").strip()
                if trans:
                    records_by_spk[spk].append(rec)

    # Sort each speaker's records canonically by sample_id
    for spk in records_by_spk:
        records_by_spk[spk].sort(key=lambda r: r["sample_id"])

    # 1. Validation Set (1,000 from s19, 1,000 from s20 -> exactly 2,000)
    print("\nAssembling validation set (s19, s20)...")
    val_records = []
    for spk in VAL_SPEAKERS:
        utts = records_by_spk[spk]
        if len(utts) != 1000:
            raise ValueError(f"Expected 1,000 utterances for {spk}, found {len(utts)}")
        for r in utts:
            rec = dict(r)
            rec["split"] = "val"
            rec["frame_count"] = 75
            rec["frame_paths"] = [
                f"E:\\LipRead_Data\\raw\\EGCLLC\\lip_images\\{spk}\\video\\mpg_6000\\{r['video_id']}\\{i}.jpg"
                for i in range(1, 76)
            ]
            val_records.append(rec)

    # 2. Test Set (1,000 from s14, 1,000 from s23 -> exactly 2,000)
    print("Assembling test set (s14, s23)...")
    test_records = []
    for spk in TEST_SPEAKERS:
        utts = records_by_spk[spk]
        if len(utts) != 1000:
            raise ValueError(f"Expected 1,000 utterances for {spk}, found {len(utts)}")
        for r in utts:
            rec = dict(r)
            rec["split"] = "test"
            rec["frame_count"] = 75
            rec["frame_paths"] = [
                f"E:\\LipRead_Data\\raw\\EGCLLC\\lip_images\\{spk}\\video\\mpg_6000\\{r['video_id']}\\{i}.jpg"
                for i in range(1, 76)
            ]
            test_records.append(rec)

    # 3. Train Set (Deterministic sample of 15,000 from 16 approved speakers)
    print("Assembling training set from 16 approved speakers...")
    train_candidates = []
    for spk in TRAIN_SPEAKERS:
        utts = records_by_spk[spk]
        for r in utts:
            rec = dict(r)
            rec["split"] = "train"
            rec["frame_count"] = 75
            rec["frame_paths"] = [
                f"E:\\LipRead_Data\\raw\\EGCLLC\\lip_images\\{spk}\\video\\mpg_6000\\{r['video_id']}\\{i}.jpg"
                for i in range(1, 76)
            ]
            train_candidates.append(rec)

    # Canonical deterministic sort by sample_id before shuffling
    train_candidates.sort(key=lambda r: r["sample_id"])
    print(f"Total valid training candidates: {len(train_candidates)}")
    if len(train_candidates) < 15000:
        raise ValueError(f"Insufficient training candidates: {len(train_candidates)} < 15000")

    # Deterministic sampling with seed 42
    rng = random.Random(42)
    rng.shuffle(train_candidates)
    train_records = train_candidates[:15000]

    # Write manifests
    print("\nWriting manifests to disk...")
    for out_path, records, label in [
        (OUT_TRAIN, train_records, "TRAIN"),
        (OUT_VAL, val_records, "VAL"),
        (OUT_TEST, test_records, "TEST"),
    ]:
        print(f"Writing {label} ({len(records):,} records) -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    print("\nManifest files written successfully!")


if __name__ == "__main__":
    create_manifests()
