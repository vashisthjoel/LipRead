"""Dataset inspection and integrity verification utility for EGCLLC in LipRead.

Inspects:
  - Directory existence (lip_images, lip_coordinates, GRID_alignments)
  - Speakers, video sequences, frame counts (min, max, mean)
  - Image dimensions, channels, corruption checks
  - Alignment files and transcript reconstruction
  - Triplet completeness (images + coordinates + alignment)
  - Vocabulary statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.data.egcllc_inspector")

DEFAULT_EGCLLC_ROOT = Path("E:/LipRead_Data/raw/EGCLLC")
SILENCE_TOKENS = {"sil", "sp", ""}


def parse_grid_alignment(align_path: Union[str, Path]) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Parses a GRID .align file into a clean sentence transcript and timed word segments.

    Format per line: <start_sample> <end_sample> <word>

    Args:
        align_path: Path to the .align file.

    Returns:
        Tuple of:
          - Full reconstructed sentence string (lowercase, stripped).
          - List of (start_idx, end_idx, word) tuples.
    """
    words: List[str] = []
    segments: List[Tuple[int, int, str]] = []

    try:
        with open(align_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_val, end_val, word = int(parts[0]), int(parts[1]), parts[2].lower()
                    segments.append((start_val, end_val, word))
                    if word not in SILENCE_TOKENS:
                        words.append(word)
    except (FileNotFoundError, OSError):
        return "", []

    sentence = " ".join(words).strip()
    return sentence, segments


def extract_speaker_id(path: Path) -> str:
    """Extracts speaker ID (e.g. 's1', 's28') from any path inside the dataset."""
    for part in path.parts:
        if part.startswith("s") and len(part) <= 4 and part[1:].isdigit():
            return part
    return "unknown"


def find_speaker_video_dirs(speaker_dir: Path) -> List[Path]:
    """Finds all video sequence directories under a speaker directory,
    handling both direct (spk/vid) and nested (spk/video/mpg_6000/vid) structures.
    """
    if not speaker_dir.exists() or not speaker_dir.is_dir():
        return []

    # Check nested: speaker/video/mpg_6000
    nested = speaker_dir / "video" / "mpg_6000"
    if nested.exists() and nested.is_dir():
        return [d for d in nested.iterdir() if d.is_dir()]

    # Check direct
    direct_subdirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
    if direct_subdirs and any(d.name not in ("video", "mpg_6000", "align") for d in direct_subdirs):
        return [d for d in direct_subdirs if d.name not in ("video", "mpg_6000", "align")]

    return []


@dataclass
class EGCLLCReport:
    """Detailed summary report produced by EGCLLCInspector."""

    root_dir: Path
    root_exists: bool
    has_images_dir: bool = False
    has_coordinates_dir: bool = False
    has_alignments_dir: bool = False
    speaker_ids: List[str] = field(default_factory=list)
    total_alignment_files: int = 0
    total_coordinate_files: int = 0
    total_image_sequences: int = 0
    total_image_frames: int = 0
    triplet_complete_count: int = 0
    missing_elements: List[str] = field(default_factory=list)

    # Frame metrics
    min_frames: int = 0
    max_frames: int = 0
    mean_frames: float = 0.0

    # Image properties
    image_resolution: Optional[Tuple[int, int]] = None
    image_channels: Optional[int] = None
    corrupted_images_detected: int = 0

    # Vocabulary & transcript statistics
    vocab_words: Set[str] = field(default_factory=set)
    vocab_chars: Set[str] = field(default_factory=set)
    sample_transcripts: List[Tuple[str, str]] = field(default_factory=list)

    def print_summary(self) -> None:
        """Prints a structured inspection report to console."""
        divider = "=" * 80
        subdivider = "-" * 80

        print("\n" + divider)
        print("LipRead - EGCLLC Dataset Inspection Report")
        print(divider)
        print(f"Dataset Root:             {self.root_dir}")
        print(f"Root Exists:              {'YES' if self.root_exists else 'NO'}")
        print(f"Subdirectories:")
        print(f"  - lip_images:           {'YES' if self.has_images_dir else 'NO'}")
        print(f"  - lip_coordinates:      {'YES' if self.has_coordinates_dir else 'NO'}")
        print(f"  - GRID_alignments:      {'YES' if self.has_alignments_dir else 'NO'}")
        print(subdivider)
        print(f"Speakers Identified:      {len(self.speaker_ids)} ({', '.join(sorted(self.speaker_ids)[:10])}...)")
        print(f"Alignment Files (.align): {self.total_alignment_files}")
        print(f"Coordinate Files (.json): {self.total_coordinate_files}")
        print(f"Image Sequences (folders):{self.total_image_sequences}")
        print(f"Image Frames (est./tot.): {self.total_image_frames}")
        print(f"Complete Triplets:        {self.triplet_complete_count}")
        print(subdivider)

        if self.image_resolution:
            print(f"Sample Image Resolution:  {self.image_resolution[0]}x{self.image_resolution[1]}")
            print(f"Sample Image Channels:    {self.image_channels}")
            print(f"Corrupted Images Sampled: {self.corrupted_images_detected}")

        if self.min_frames > 0:
            print(f"Sequence Frame Stats:     Min={self.min_frames}, Max={self.max_frames}, Mean={self.mean_frames:.1f}")

        if self.vocab_words:
            print(f"\nVocabulary Statistics:")
            print(f"  - Unique Words:         {len(self.vocab_words)}")
            print(f"  - Character Set:        {''.join(sorted(list(self.vocab_chars)))}")

        if self.sample_transcripts:
            print(f"\nSample Alignment Transcripts:")
            for sample_id, text in self.sample_transcripts[:5]:
                print(f"  [{sample_id}] -> \"{text}\"")

        if self.missing_elements:
            print(f"\nMissing Elements / Issues:")
            for m in self.missing_elements:
                print(f"  [X] {m}")

        print(divider + "\n")


class EGCLLCInspector:
    """Non-destructive scanner and validator for EGCLLC dataset installations."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir).resolve() if root_dir else DEFAULT_EGCLLC_ROOT
        self.images_dir = self.root_dir / "lip_images"
        self.coords_dir = self.root_dir / "lip_coordinates"
        self.aligns_dir = self.root_dir / "GRID_alignments"

    def inspect(self, sample_size: int = 50) -> EGCLLCReport:
        """Inspects the dataset structure without loading everything into RAM.

        Args:
            sample_size: Number of sample sequences to inspect deeply (image loading, etc.).

        Returns:
            EGCLLCReport containing structured inspection metrics.
        """
        if not self.root_dir.exists():
            return EGCLLCReport(
                root_dir=self.root_dir,
                root_exists=False,
                missing_elements=[f"Root directory does not exist: {self.root_dir}"],
            )

        has_images = self.images_dir.exists()
        has_coords = self.coords_dir.exists()
        has_aligns = self.aligns_dir.exists()

        missing: List[str] = []
        if not has_images:
            missing.append("lip_images directory is missing or not yet extracted.")
        if not has_coords:
            missing.append("lip_coordinates directory is missing or not yet extracted.")
        if not has_aligns:
            missing.append("GRID_alignments directory is missing or not yet extracted.")

        # Identify speakers across available directories
        speakers: Set[str] = set()
        for base in [self.images_dir, self.coords_dir, self.aligns_dir]:
            if base.exists():
                for item in base.iterdir():
                    if item.is_dir() and item.name.startswith("s"):
                        speakers.add(item.name)

        sorted_speakers = sorted(list(speakers))

        # Count alignment files
        alignment_files: List[Path] = []
        if has_aligns:
            alignment_files = list(self.aligns_dir.rglob("*.align"))

        # Count coordinate files
        coord_files: List[Path] = []
        if has_coords:
            coord_files = list(self.coords_dir.rglob("*.json"))

        # Count image sequences
        image_seqs: List[Path] = []
        if has_images:
            for spk in sorted_speakers:
                spk_dir = self.images_dir / spk
                if spk_dir.exists():
                    image_seqs.extend(find_speaker_video_dirs(spk_dir))

        # Build index of available IDs per modality: speaker_id / video_id
        align_keys: Set[str] = set()
        for p in alignment_files:
            video_id = p.stem
            spk = extract_speaker_id(p)
            align_keys.add(f"{spk}/{video_id}")

        coord_keys: Set[str] = set()
        for p in coord_files:
            video_id = p.stem
            spk = extract_speaker_id(p)
            coord_keys.add(f"{spk}/{video_id}")

        image_keys: Set[str] = set()
        for p in image_seqs:
            video_id = p.name
            spk = extract_speaker_id(p)
            image_keys.add(f"{spk}/{video_id}")

        # Complete triplets
        common_triplets = align_keys & coord_keys & image_keys if image_keys else align_keys & coord_keys

        # Detailed sampling for image resolution, frames, vocabulary
        sample_frame_counts: List[int] = []
        sample_res: Optional[Tuple[int, int]] = None
        sample_ch: Optional[int] = None
        corrupted_count = 0
        vocab_words: Set[str] = set()
        vocab_chars: Set[str] = set()
        sample_transcripts: List[Tuple[str, str]] = []

        # Parse sample alignments
        for i, align_p in enumerate(alignment_files[:sample_size]):
            video_id = align_p.stem
            spk = extract_speaker_id(align_p)
            sentence, _ = parse_grid_alignment(align_p)
            if sentence:
                sample_transcripts.append((f"{spk}/{video_id}", sentence))
                for w in sentence.split():
                    vocab_words.add(w)
                    for c in w:
                        vocab_chars.add(c)

        # Inspect image sequences (if extracted)
        total_frames_est = 0
        if image_seqs:
            for seq_dir in image_seqs[:sample_size]:
                frames = sorted(
                    list(seq_dir.glob("*.jpg")),
                    key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem,
                )
                sample_frame_counts.append(len(frames))

                if frames and sample_res is None:
                    # Test read first image
                    img = cv2.imread(str(frames[0]))
                    if img is not None:
                        sample_res = (img.shape[0], img.shape[1])
                        sample_ch = img.shape[2] if img.ndim == 3 else 1
                    else:
                        corrupted_count += 1

            if sample_frame_counts:
                total_frames_est = int(np.mean(sample_frame_counts) * len(image_seqs))

        min_f = min(sample_frame_counts) if sample_frame_counts else 0
        max_f = max(sample_frame_counts) if sample_frame_counts else 0
        mean_f = float(np.mean(sample_frame_counts)) if sample_frame_counts else 0.0

        return EGCLLCReport(
            root_dir=self.root_dir,
            root_exists=True,
            has_images_dir=has_images,
            has_coordinates_dir=has_coords,
            has_alignments_dir=has_aligns,
            speaker_ids=sorted_speakers,
            total_alignment_files=len(alignment_files),
            total_coordinate_files=len(coord_files),
            total_image_sequences=len(image_seqs),
            total_image_frames=total_frames_est,
            triplet_complete_count=len(common_triplets),
            missing_elements=missing,
            min_frames=min_f,
            max_frames=max_f,
            mean_frames=mean_f,
            image_resolution=sample_res,
            image_channels=sample_ch,
            corrupted_images_detected=corrupted_count,
            vocab_words=vocab_words,
            vocab_chars=vocab_chars,
            sample_transcripts=sample_transcripts,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect EGCLLC dataset status and contents")
    parser.add_argument("--root", "-r", type=str, default=None, help="Custom EGCLLC root directory")
    parser.add_argument("--sample-size", "-s", type=int, default=50, help="Sample size for deep checks")
    args = parser.parse_args()

    inspector = EGCLLCInspector(root_dir=args.root)
    report = inspector.inspect(sample_size=args.sample_size)
    report.print_summary()

    return 0 if (report.has_images_dir and report.has_alignments_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
