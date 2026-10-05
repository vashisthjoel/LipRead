"""Command-line interface for EGCLLC dataset management in LipRead.

Commands:
    python -m src.data.egcllc_cli inspect
    python -m src.data.egcllc_cli build-index
    python -m src.data.egcllc_cli create-splits
    python -m src.data.egcllc_cli verify-sample [--sample-id s1/bbaf2n]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

from src.data.egcllc_adapter import EGCLLCAdapter
from src.data.egcllc_dataset import EGCLLCDataset, EGCLLCIndexBuilder
from src.data.egcllc_inspector import EGCLLCInspector
from src.data.paths import get_metadata_dir
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.data.egcllc_cli")


def cmd_inspect(args: argparse.Namespace) -> int:
    """Executes dataset inspection."""
    inspector = EGCLLCInspector(root_dir=args.root)
    report = inspector.inspect(sample_size=args.sample_size)
    report.print_summary()
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    """Builds full dataset metadata index and speaker-aware split manifests."""
    print("\n" + "=" * 70)
    print("LipRead - Building EGCLLC Metadata Index & Splits")
    print("=" * 70)
    try:
        builder = EGCLLCIndexBuilder(
            dataset_root=args.root,
            metadata_dir=args.metadata_dir,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        summary = builder.build_and_save_index()

        print(f"\nIndex Generation Complete:")
        print(f"Total Speakers:   {summary['total_speakers']}")
        print(f"Total Samples:    {summary['total_samples']}")
        print(f"Train Split:      {summary['splits']['train']['sample_count']} samples ({len(summary['splits']['train']['speakers'])} speakers)")
        print(f"Val Split:        {summary['splits']['val']['sample_count']} samples ({len(summary['splits']['val']['speakers'])} speakers)")
        print(f"Test Split:       {summary['splits']['test']['sample_count']} samples ({len(summary['splits']['test']['speakers'])} speakers)")
        print("=" * 70 + "\n")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Failed building index: {exc}\n", file=sys.stderr)
        return 1


def cmd_verify_sample(args: argparse.Namespace) -> int:
    """Loads a single real sample through the adapter and verifies tensor and transcript."""
    print("\n" + "=" * 70)
    print("LipRead - EGCLLC Real Sample Verification")
    print("=" * 70)

    meta_file = get_metadata_dir() / "egcllc_index.jsonl"
    if not meta_file.exists():
        meta_file = get_metadata_dir() / "train.jsonl"

    if not meta_file.exists():
        print(f"[ERROR] No index file found at {meta_file}. Run 'build-index' first.", file=sys.stderr)
        return 1

    target_sample = None
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if args.sample_id:
                if item["sample_id"] == args.sample_id or item["video_id"] == args.sample_id:
                    target_sample = item
                    break
            else:
                # Default to first sample with frame paths if available
                if item.get("frame_paths"):
                    target_sample = item
                    break
                elif target_sample is None:
                    target_sample = item

    if target_sample is None:
        print(f"[ERROR] Sample '{args.sample_id}' not found in index.", file=sys.stderr)
        return 1

    sample_id = target_sample["sample_id"]
    spk = target_sample["speaker_id"]
    vid = target_sample["video_id"]
    transcript = target_sample["transcript"]
    frame_paths = target_sample.get("frame_paths", [])

    print(f"Sample ID:        {sample_id}")
    print(f"Speaker ID:       {spk}")
    print(f"Video ID:         {vid}")
    print(f"Transcript:       \"{transcript}\"")
    print(f"Frame Count:      {len(frame_paths)}")

    adapter = EGCLLCAdapter()
    tensor = adapter.process_frame_paths(frame_paths)

    print("-" * 70)
    print(f"Tensor Shape:     {tuple(tensor.shape)} [T, C, H, W]")
    print(f"Tensor Dtype:     {tensor.dtype}")
    print(f"Value Range:      [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    print(f"Value Mean / Std: {tensor.mean().item():.3f} / {tensor.std().item():.3f}")
    print("=" * 70 + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EGCLLC Dataset Management CLI for LipRead",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect EGCLLC dataset structure")
    p_inspect.add_argument("--root", "-r", type=str, default=None, help="EGCLLC root directory")
    p_inspect.add_argument("--sample-size", "-s", type=int, default=50, help="Sampling size")
    p_inspect.set_defaults(func=cmd_inspect)

    # build-index
    p_index = subparsers.add_parser("build-index", help="Build metadata index and splits")
    p_index.add_argument("--root", "-r", type=str, default=None, help="EGCLLC root directory")
    p_index.add_argument("--metadata-dir", "-m", type=str, default=None, help="Output metadata directory")
    p_index.add_argument("--seed", type=int, default=42, help="Random seed for speaker splitting")
    p_index.add_argument("--val-ratio", type=float, default=0.15, help="Validation speaker fraction")
    p_index.add_argument("--test-ratio", type=float, default=0.15, help="Test speaker fraction")
    p_index.set_defaults(func=cmd_build_index)

    # create-splits (alias for build-index)
    p_splits = subparsers.add_parser("create-splits", help="Generate speaker-aware splits")
    p_splits.add_argument("--root", "-r", type=str, default=None, help="EGCLLC root directory")
    p_splits.add_argument("--metadata-dir", "-m", type=str, default=None, help="Output metadata directory")
    p_splits.add_argument("--seed", type=int, default=42, help="Random seed for speaker splitting")
    p_splits.add_argument("--val-ratio", type=float, default=0.15, help="Validation speaker fraction")
    p_splits.add_argument("--test-ratio", type=float, default=0.15, help="Test speaker fraction")
    p_splits.set_defaults(func=cmd_build_index)

    # verify-sample
    p_verify = subparsers.add_parser("verify-sample", help="Verify and process a single real sample")
    p_verify.add_argument("--sample-id", "-s", type=str, default=None, help="Specific sample ID to verify")
    p_verify.set_defaults(func=cmd_verify_sample)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
