"""Configuration schemas and data structures for the LRS2 dataset.

NOTE: This module defines formal data models and configuration contracts
for sentence-level lip reading (LRS2). It does NOT implement an active dataset
loader or ingest pipelines yet, because the actual dataset filesystem layout
and annotations must be inspected and verified upon physical acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.config import METADATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


class LRS2Split(str, Enum):
    """LRS2 standard dataset splits."""

    PRETRAIN = "pretrain"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class VocabularyConfig:
    """Character vocabulary configuration for CTC decoding."""

    characters: str = " abcdefghijklmnopqrstuvwxyz'"
    blank_token: str = "<blank>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Duplicate characters found in vocabulary definition.")

    @property
    def blank_idx(self) -> int:
        """CTC blank token index (always 0)."""
        return 0

    @property
    def unk_idx(self) -> int:
        """Unknown token index (last position)."""
        return len(self.characters) + 1

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including CTC blank and unk tokens."""
        # 0: blank, 1..N: characters, N+1: unk
        return len(self.characters) + 2

    def char_to_id(self, char: str) -> int:
        """Maps a single character to its vocabulary index."""
        normalized = char.lower()
        idx = self.characters.find(normalized)
        if idx != -1:
            return idx + 1  # 1-indexed because 0 is blank
        return self.unk_idx

    def id_to_char(self, idx: int) -> str:
        """Maps an index back to its corresponding character."""
        if idx == self.blank_idx:
            return ""
        if 1 <= idx <= len(self.characters):
            return self.characters[idx - 1]
        return ""

    def encode(self, text: str) -> List[int]:
        """Encodes an input string into token IDs."""
        return [self.char_to_id(c) for c in text.strip()]

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs into a raw string."""
        return "".join([self.id_to_char(i) for i in token_ids])


@dataclass
class LRS2SampleMetadata:
    """Represents metadata for a single video utterance in LRS2."""

    sample_id: str
    video_rel_path: str
    transcription: str
    split: LRS2Split
    duration_frames: Optional[int] = None
    speaker_id: Optional[str] = None


@dataclass
class LRS2DatasetConfig:
    """High-level configuration for LRS2 dataset paths and preprocessing specs."""

    dataset_name: str = "LRS2"
    raw_dir: Path = field(default_factory=lambda: RAW_DATA_DIR / "lrs2_v1")
    processed_dir: Path = field(default_factory=lambda: PROCESSED_DATA_DIR / "lrs2_crops")
    metadata_dir: Path = field(default_factory=lambda: METADATA_DIR)

    # Video spatiotemporal parameters
    target_image_size: Tuple[int, int] = (88, 88)
    fps: int = 25
    max_sequence_length: int = 150  # ~6.0 seconds max at 25 fps
    channels: int = 1  # Grayscale for lip movements

    # Standard split filenames expected from LRS2 distribution
    split_files: Dict[LRS2Split, str] = field(
        default_factory=lambda: {
            LRS2Split.PRETRAIN: "pretrain.txt",
            LRS2Split.TRAIN: "train.txt",
            LRS2Split.VAL: "val.txt",
            LRS2Split.TEST: "test.txt",
        }
    )

    vocab: VocabularyConfig = field(default_factory=VocabularyConfig)
