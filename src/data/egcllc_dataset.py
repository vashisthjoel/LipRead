"""PyTorch Dataset and Index Builder for EGCLLC in LipRead.

Provides:
  - EGCLLCSample dataclass
  - EGCLLCIndexBuilder for reproducible, speaker-aware split generation
  - EGCLLCDataset with lazy frame loading, vocabulary encoding, and optional tensor caching
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset_config import VocabularyConfig
from src.data.egcllc_adapter import EGCLLCAdapter
from src.data.egcllc_inspector import DEFAULT_EGCLLC_ROOT, parse_grid_alignment
from src.data.paths import get_metadata_dir, get_processed_data_dir
from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.data.egcllc_dataset")


@dataclass
class EGCLLCSample:
    """Represents a single utterance sample in the EGCLLC dataset."""

    sample_id: str
    speaker_id: str
    video_id: str
    frame_paths: List[str]
    transcript: str
    frame_count: int
    alignment_path: Optional[str] = None
    coordinate_path: Optional[str] = None
    split: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EGCLLCSample:
        return cls(**data)


class EGCLLCIndexBuilder:
    """Discovers, parses, and indexes EGCLLC into reproducible speaker-aware JSONL manifests."""

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        metadata_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """Initializes index builder.

        Args:
            dataset_root: Path to EGCLLC root directory (containing lip_images, etc.).
            metadata_dir: Target output directory for JSONL index and split manifests.
            seed: Random seed for deterministic speaker assignment.
            val_ratio: Proportion of speakers assigned to validation set.
            test_ratio: Proportion of speakers assigned to test set.
        """
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else DEFAULT_EGCLLC_ROOT
        self.metadata_dir = Path(metadata_dir).resolve() if metadata_dir else get_metadata_dir()
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.images_dir = self.dataset_root / "lip_images"
        self.coords_dir = self.dataset_root / "lip_coordinates"
        self.aligns_dir = self.dataset_root / "GRID_alignments"

    def build_and_save_index(self) -> Dict[str, Any]:
        """Discovers all valid samples, assigns speaker-aware splits, and writes JSONL manifests."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Scanning EGCLLC dataset at %s...", self.dataset_root)

        # 1. Discover all speakers from alignments and images
        speakers: Set[str] = set()
        if self.aligns_dir.exists():
            for p in self.aligns_dir.iterdir():
                if p.is_dir() and p.name.startswith("s"):
                    speakers.add(p.name)

        if self.images_dir.exists():
            for p in self.images_dir.iterdir():
                if p.is_dir() and p.name.startswith("s"):
                    speakers.add(p.name)

        sorted_speakers = sorted(list(speakers))
        if not sorted_speakers:
            raise RuntimeError(f"No speaker directories found in {self.dataset_root}")

        # 2. Partition speakers deterministically (NO speaker leakage)
        # Using canonical LipNet evaluation benchmark or deterministic seed-based partition
        rng = random.Random(self.seed)
        shuffled = sorted_speakers.copy()
        rng.shuffle(shuffled)

        n_test = max(1, int(round(len(sorted_speakers) * self.test_ratio)))
        n_val = max(1, int(round(len(sorted_speakers) * self.val_ratio)))

        test_speakers = set(sorted(shuffled[:n_test]))
        val_speakers = set(sorted(shuffled[n_test : n_test + n_val]))
        train_speakers = set(sorted(shuffled[n_test + n_val :]))

        # Verify disjointness
        assert test_speakers.isdisjoint(val_speakers)
        assert test_speakers.isdisjoint(train_speakers)
        assert val_speakers.isdisjoint(train_speakers)

        speaker_to_split: Dict[str, str] = {}
        for s in train_speakers:
            speaker_to_split[s] = "train"
        for s in val_speakers:
            speaker_to_split[s] = "val"
        for s in test_speakers:
            speaker_to_split[s] = "test"

        logger.info(
            "Speaker partition: %d Train (%s...), %d Val (%s), %d Test (%s)",
            len(train_speakers),
            list(sorted(train_speakers))[:5],
            len(val_speakers),
            list(sorted(val_speakers)),
            len(test_speakers),
            list(sorted(test_speakers)),
        )

        # 3. Discover and index samples
        samples: List[EGCLLCSample] = []
        samples_by_split: Dict[str, List[EGCLLCSample]] = {"train": [], "val": [], "test": []}

        # Scan each speaker
        for spk_idx, spk in enumerate(sorted_speakers):
            split_name = speaker_to_split.get(spk, "train")

            # Alignments directory for this speaker (may be s1/align or s1/)
            spk_align_dir = self.aligns_dir / spk
            if (spk_align_dir / "align").exists():
                spk_align_dir = spk_align_dir / "align"

            align_files: Dict[str, str] = {}
            if spk_align_dir.exists():
                try:
                    for entry in os.scandir(spk_align_dir):
                        if entry.name.endswith(".align"):
                            align_files[entry.name[:-6]] = entry.path
                except OSError:
                    pass

            # Coordinates directory
            spk_coord_dir = self.coords_dir / spk
            coord_files: Dict[str, str] = {}
            if spk_coord_dir.exists():
                try:
                    for entry in os.scandir(spk_coord_dir):
                        if entry.name.endswith(".json"):
                            coord_files[entry.name[:-5]] = entry.path
                except OSError:
                    pass

            # Images directory
            spk_img_dir = self.images_dir / spk
            img_vid_map: Dict[str, str] = {}
            if spk_img_dir.exists():
                nested_dir = spk_img_dir / "video" / "mpg_6000"
                target_parent = nested_dir if nested_dir.exists() else spk_img_dir
                try:
                    for entry in os.scandir(target_parent):
                        if entry.is_dir() and entry.name not in ("video", "mpg_6000"):
                            img_vid_map[entry.name] = entry.path
                except OSError:
                    pass

            # Build sequence list: union of alignments, coords, and image dirs
            video_ids = sorted(list(set(align_files.keys()) | set(coord_files.keys()) | set(img_vid_map.keys())))

            # Parallel parse alignment transcripts for this speaker
            def _parse_transcript(vid: str) -> Tuple[str, str]:
                a_p = align_files.get(vid)
                if not a_p:
                    return vid, ""
                s, _ = parse_grid_alignment(a_p)
                return vid, s

            with ThreadPoolExecutor(max_workers=16) as pool:
                transcripts = dict(pool.map(_parse_transcript, video_ids))

            spk_samples: List[EGCLLCSample] = []
            for vid in video_ids:
                transcript = transcripts.get(vid, "")
                align_path = align_files.get(vid)
                coord_path = coord_files.get(vid)

                # Locate frame paths if images are present
                frame_paths: List[str] = []
                vid_img_path = img_vid_map.get(vid)
                if vid_img_path:
                    try:
                        img_files = [
                            f
                            for f in os.listdir(vid_img_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        ]
                        img_files.sort(
                            key=lambda name: int(name.rsplit(".", 1)[0])
                            if name.rsplit(".", 1)[0].isdigit()
                            else name
                        )
                        frame_paths = [os.path.join(vid_img_path, name) for name in img_files]
                    except OSError:
                        pass

                sample = EGCLLCSample(
                    sample_id=f"{spk}/{vid}",
                    speaker_id=spk,
                    video_id=vid,
                    frame_paths=frame_paths,
                    transcript=transcript,
                    frame_count=len(frame_paths),
                    alignment_path=align_path,
                    coordinate_path=coord_path,
                    split=split_name,
                )

                samples.append(sample)
                samples_by_split[split_name].append(sample)
                spk_samples.append(sample)

            logger.info(
                "Indexed speaker %s (%d/%d): %d samples (split=%s)",
                spk,
                spk_idx + 1,
                len(sorted_speakers),
                len(spk_samples),
                split_name,
            )

        # 4. Save JSONL index files
        all_index_path = self.metadata_dir / "egcllc_index.jsonl"
        with open(all_index_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")

        split_paths: Dict[str, Path] = {}
        for s_name in ("train", "val", "test"):
            p = self.metadata_dir / f"{s_name}.jsonl"
            split_paths[s_name] = p
            with open(p, "w", encoding="utf-8") as f:
                for s in samples_by_split[s_name]:
                    f.write(json.dumps(s.to_dict()) + "\n")

        # Save human-readable summary
        summary = {
            "dataset_root": str(self.dataset_root),
            "seed": self.seed,
            "total_speakers": len(sorted_speakers),
            "total_samples": len(samples),
            "splits": {
                "train": {
                    "speakers": sorted(list(train_speakers)),
                    "sample_count": len(samples_by_split["train"]),
                },
                "val": {
                    "speakers": sorted(list(val_speakers)),
                    "sample_count": len(samples_by_split["val"]),
                },
                "test": {
                    "speakers": sorted(list(test_speakers)),
                    "sample_count": len(samples_by_split["test"]),
                },
            },
        }

        with open(self.metadata_dir / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Saved index: %d total samples (train=%d, val=%d, test=%d) to %s",
            len(samples),
            len(samples_by_split["train"]),
            len(samples_by_split["val"]),
            len(samples_by_split["test"]),
            self.metadata_dir,
        )

        return summary


class EGCLLCDataset(Dataset):
    """PyTorch Dataset loading EGCLLC lip reading sequences on-demand."""

    def __init__(
        self,
        split: str = "train",
        metadata_file: Optional[Union[str, Path]] = None,
        use_cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        filter_extracted: bool = False,
    ):
        """Initializes the dataset.

        Args:
            split: Target split name ('train', 'val', or 'test').
            metadata_file: Path to specific JSONL file. Defaults to metadata/<split>.jsonl.
            use_cache: If True, caches preprocessed tensors to disk to avoid repeated JPEG loading.
            cache_dir: Target directory for tensor cache. Defaults to E:\\LipRead_Data\\processed\\egcllc.
            max_samples: Optional limit on total samples loaded (for debugging/smoke tests).
            filter_extracted: If True, filters records to only those with frames extracted on disk.
        """
        self.split = split
        self.use_cache = use_cache
        self.filter_extracted = filter_extracted

        if metadata_file is not None:
            self.meta_path = Path(metadata_file).resolve()
        else:
            self.meta_path = get_metadata_dir() / f"{split}.jsonl"

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata split manifest not found: {self.meta_path}")

        # Setup cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = get_processed_data_dir() / "egcllc"

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.vocab = VocabularyConfig()
        self.adapter = EGCLLCAdapter()
        self.raw_lip_images_dir = Path(r"E:\LipRead_Data\raw\EGCLLC\lip_images")

        # Discover extracted speakers once to avoid thousands of disk stat calls
        extracted_speakers = set()
        if self.raw_lip_images_dir.exists():
            try:
                extracted_speakers = {d.name for d in self.raw_lip_images_dir.iterdir() if d.is_dir()}
            except Exception:
                pass

        # Load metadata records lazily into memory
        self.samples: List[Dict[str, Any]] = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if self.filter_extracted:
                        if item["speaker_id"] not in extracted_speakers:
                            continue
                    self.samples.append(item)
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break

        logger.debug("Loaded %d samples from %s (filter_extracted=%s)", len(self.samples), self.meta_path.name, self.filter_extracted)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cache_path(self, speaker_id: str, video_id: str) -> Path:
        return self.cache_dir / f"{speaker_id}_{video_id}.pt"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        spk = sample["speaker_id"]
        vid = sample["video_id"]
        transcript = sample["transcript"]
        frame_paths = sample.get("frame_paths", [])

        # Direct deterministic path construction if not pre-populated in jsonl
        if not frame_paths:
            raw_vid_dir = self.raw_lip_images_dir / spk / "video" / "mpg_6000" / vid
            frame_paths = [str(raw_vid_dir / f"{i}.jpg") for i in range(1, 76)]

        # Check tensor cache if enabled
        tensor: Optional[torch.Tensor] = None
        if self.use_cache:
            cache_file = self._get_cache_path(spk, vid)
            if cache_file.exists():
                try:
                    tensor = torch.load(cache_file, weights_only=True)
                except Exception as exc:
                    logger.debug("Failed reading cache %s: %s", cache_file, exc)

        # Process frames via adapter if not retrieved from cache
        if tensor is None:
            tensor = self.adapter.process_frame_paths(frame_paths)
            if self.use_cache:
                try:
                    cache_file = self._get_cache_path(spk, vid)
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(tensor, cache_file)
                except Exception as exc:
                    logger.debug("Could not write cache file %s: %s", cache_file, exc)

        # Encode transcript to CTC token IDs
        token_ids = self.vocab.encode(transcript)
        tokens_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "frames": tensor,  # Shape: [T, C, H, W] -> [75, 1, 88, 88]
            "tokens": tokens_tensor,  # Shape: [L]
            "transcript": transcript,
            "sample_id": sample["sample_id"],
            "speaker_id": spk,
            "video_id": vid,
            "frame_count": len(frame_paths),
            "token_count": len(token_ids),
        }
