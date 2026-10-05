"""Dataset inspection utility for LRS2 in LipRead.

Inspects raw data storage, verifies whether an authentic LRS2 distribution
is present, and reports structure, video counts, splits, and missing artifacts
without modifying files or preloading heavy videos.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.data.paths import get_raw_data_dir
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.data.inspector")

CANONICAL_SPLIT_FILES = ("train.txt", "val.txt", "test.txt", "pretrain.txt")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
TRANSCRIPT_EXTENSIONS = {".txt", ".tsv", ".csv"}


@dataclass
class LRS2InspectionReport:
    """Structured report documenting the contents of the raw dataset directory."""

    raw_dir: Path
    exists: bool
    is_empty: bool
    is_lrs2_detected: bool
    total_videos: int = 0
    total_transcripts: int = 0
    extension_counts: Dict[str, int] = field(default_factory=dict)
    discovered_splits: Dict[str, int] = field(default_factory=dict)
    sample_video_paths: List[str] = field(default_factory=list)
    missing_elements: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Prints a structured console summary report."""
        divider = "=" * 80
        subdivider = "-" * 80

        print("\n" + divider)
        print("LipRead - LRS2 Dataset Inspection Report")
        print(divider)
        print(f"Target Directory:        {self.raw_dir}")
        print(f"Directory Exists:        {'YES' if self.exists else 'NO'}")
        print(f"Directory Empty:         {'YES' if self.is_empty else 'NO'}")
        print(f"LRS2 Detected:           {'YES' if self.is_lrs2_detected else 'NO'}")
        print(subdivider)
        print(f"Total Video Files:       {self.total_videos}")
        print(f"Total Transcript Files:  {self.total_transcripts}")

        if self.extension_counts:
            ext_str = ", ".join(f"{ext}: {cnt}" for ext, cnt in sorted(self.extension_counts.items()))
            print(f"File Extensions:         {ext_str}")

        if self.discovered_splits:
            print("\nDiscovered Splits:")
            for split_name, count in sorted(self.discovered_splits.items()):
                print(f"  - {split_name:<16}: {count} utterance references")
        else:
            print("Discovered Splits:       None")

        if self.sample_video_paths:
            print("\nSample Video Paths (first 5):")
            for p in self.sample_video_paths[:5]:
                print(f"  - {p}")

        if self.missing_elements:
            print("\nMissing Required Elements:")
            for m in self.missing_elements:
                print(f"  [X] {m}")

        if self.notes:
            print("\nActionable Next Steps:")
            for n in self.notes:
                print(f"  [!] {n}")

        print(divider + "\n")


class LRS2Inspector:
    """Non-destructive scanner for LRS2 dataset hierarchies."""

    def __init__(self, raw_dir: Optional[Union[str, Path]] = None):
        """Initializes the inspector with target raw data directory.

        Args:
            raw_dir: Path to raw dataset root. Defaults to E:\\LipRead_Data\\raw.
        """
        self.raw_dir = Path(raw_dir).resolve() if raw_dir else get_raw_data_dir()

    def inspect(self) -> LRS2InspectionReport:
        """Scans the directory structure and returns an inspection report."""
        if not self.raw_dir.exists():
            return LRS2InspectionReport(
                raw_dir=self.raw_dir,
                exists=False,
                is_empty=True,
                is_lrs2_detected=False,
                missing_elements=["Raw directory does not exist on disk."],
                notes=[f"Ensure storage volume is accessible and create {self.raw_dir}."],
            )

        # List all immediate and recursive items shallowly/safely
        all_files: List[Path] = []
        try:
            for item in self.raw_dir.rglob("*"):
                if item.is_file():
                    all_files.append(item)
                    if len(all_files) >= 50000:
                        logger.warning("Large number of files detected. Capping initial scan at 50,000 items.")
                        break
        except Exception as exc:
            logger.error("Error scanning raw directory: %s", exc)

        is_empty = len(all_files) == 0

        if is_empty:
            return LRS2InspectionReport(
                raw_dir=self.raw_dir,
                exists=True,
                is_empty=True,
                is_lrs2_detected=False,
                missing_elements=[
                    "No video files (.mp4) found in raw storage.",
                    "No utterance text transcriptions (.txt) found.",
                    "No standard split manifests (train.txt, val.txt, test.txt) found.",
                    "No main/ or pretrain/ subdirectories found.",
                ],
                notes=[
                    "LRS2 dataset has not yet been placed into the raw data storage directory.",
                    f"Please place official LRS2 archives or extracted folders inside: {self.raw_dir}",
                    "Expected layout example: E:\\LipRead_Data\\raw\\mains\\<speaker_id>\\<utterance_id>.mp4",
                    "Do NOT fabricate mock samples or invent fake dataset files.",
                ],
            )

        # Categorize discovered files
        ext_counter: Counter[str] = Counter()
        video_files: List[Path] = []
        transcript_files: List[Path] = []
        discovered_splits: Dict[str, int] = {}

        for f in all_files:
            ext = f.suffix.lower()
            ext_counter[ext] += 1
            if ext in VIDEO_EXTENSIONS:
                video_files.append(f)
            elif ext in TRANSCRIPT_EXTENSIONS:
                transcript_files.append(f)
                if f.name.lower() in CANONICAL_SPLIT_FILES:
                    try:
                        with open(f, "r", encoding="utf-8", errors="ignore") as sf:
                            lines = [line.strip() for line in sf if line.strip()]
                            discovered_splits[f.name] = len(lines)
                    except Exception:
                        discovered_splits[f.name] = 0

        # Detection heuristics for genuine LRS2
        has_videos = len(video_files) > 0
        has_splits = len(discovered_splits) > 0
        is_lrs2 = has_videos and (has_splits or len(transcript_files) > 0)

        missing: List[str] = []
        if not has_videos:
            missing.append("No video files (.mp4) found.")
        if not discovered_splits:
            missing.append("No canonical split text files (train.txt, val.txt, test.txt) found.")
        if len(transcript_files) == 0:
            missing.append("No transcript text files found.")

        notes: List[str] = []
        if not is_lrs2:
            notes.append("Files found in raw directory do not match LRS2 conventions.")
            notes.append(f"Target location: {self.raw_dir}")
        else:
            notes.append("LRS2 dataset structure recognized. Ready for metadata indexing.")

        sample_paths = [str(p.relative_to(self.raw_dir)) for p in video_files[:5]]

        return LRS2InspectionReport(
            raw_dir=self.raw_dir,
            exists=True,
            is_empty=False,
            is_lrs2_detected=is_lrs2,
            total_videos=len(video_files),
            total_transcripts=len(transcript_files),
            extension_counts=dict(ext_counter),
            discovered_splits=discovered_splits,
            sample_video_paths=sample_paths,
            missing_elements=missing,
            notes=notes,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LRS2 raw dataset storage")
    parser.add_argument("--path", "-p", type=str, default=None, help="Custom raw data directory to inspect")
    args = parser.parse_args()

    inspector = LRS2Inspector(raw_dir=args.path)
    report = inspector.inspect()
    report.print_summary()

    return 0 if report.is_lrs2_detected else 1


if __name__ == "__main__":
    sys.exit(main())
