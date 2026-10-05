"""Command-line interface for video-to-tensor preprocessing in LipRead.

Usage:
    python -m src.preprocessing.cli --video path/to/video.mp4 [--output path/to/output.pt]

Examples:
    python -m src.preprocessing.cli --video sample.mp4
    python -m src.preprocessing.cli --video sample.mp4 --output E:\\LipRead_Data\\processed\\sample.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from src.preprocessing.pipeline import LipPreprocessingPipeline
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.preprocessing.cli")


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="LipRead - Video Preprocessing & Mouth ROI Extraction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Path to the input video file (.mp4, .avi, .mov, etc.)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save the processed PyTorch tensor (.pt)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of input frames to process",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    """Executes preprocessing from parsed arguments."""
    video_path = Path(args.video).resolve()

    if not video_path.exists():
        print(f"\n[ERROR] Video file not found: {video_path}\n", file=sys.stderr)
        return 1

    if not video_path.is_file():
        print(f"\n[ERROR] Path is not a regular file: {video_path}\n", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("LipRead - Video Preprocessing Pipeline")
    print("=" * 70)
    print(f"Target Video:    {video_path}")
    if args.output:
        print(f"Target Output:   {args.output}")
    print("-" * 70)

    try:
        with LipPreprocessingPipeline() as pipeline:
            result = pipeline.process_video(video_path=video_path, max_frames=args.max_frames)

        print("\n" + result.summary())

        if not result.success or result.tensor is None:
            print(f"\n[FAILURE] Processing did not complete: {result.error_message}\n")
            return 1

        # Tensor statistics
        t = result.tensor
        print(f"Tensor Type:            {t.dtype}")
        print(f"Tensor Shape:           {tuple(t.shape)} [T, C, H, W]")
        print(f"Value Range:            [{t.min().item():.3f}, {t.max().item():.3f}]")
        print(f"Value Mean / Std:       {t.mean().item():.3f} / {t.std().item():.3f}")

        # Optional tensor serialization
        if args.output:
            out_path = Path(args.output).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(t, str(out_path))
            file_size_kb = out_path.stat().st_size / 1024
            print(f"\nSaved processed tensor to: {out_path} ({file_size_kb:.1f} KB)")

        print("=" * 70 + "\n")
        return 0

    except Exception as exc:
        print(f"\n[EXCEPTION] Preprocessing failed: {exc}\n", file=sys.stderr)
        return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
