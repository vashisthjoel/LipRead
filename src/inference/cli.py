"""Command-line interface for LipRead inference.

Performs end-to-end transcription of a video file using a trained checkpoint.

Usage:
    python -m src.inference.cli --video path/to/video.mp4 --checkpoint E:\\LipRead_Data\\checkpoints\\best_model.pt
    python -m src.inference.cli --video sample.mp4 --method beam --beam-width 20
    python -m src.inference.cli --video sample.mp4 --no-gpu
    python -m src.inference.cli --tensor path/to/frames.pt --checkpoint best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.inference.cli")


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for inference CLI."""
    parser = argparse.ArgumentParser(
        description="LipRead — Lip-Reading Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input (mutually exclusive: video file or pre-processed tensor)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        "-v",
        type=str,
        help="Path to input video file (.mp4, .avi, .mov, etc.)",
    )
    input_group.add_argument(
        "--tensor",
        "-t",
        type=str,
        help="Path to pre-processed PyTorch tensor (.pt) with shape [T, C, H, W].",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint (.pt). "
            "Defaults to E:\\LipRead_Data\\checkpoints\\best_model.pt if not specified."
        ),
    )

    # Decoding
    parser.add_argument(
        "--method",
        "-m",
        choices=["greedy", "beam"],
        default="greedy",
        help="CTC decoding algorithm.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
        help="Beam width (only relevant with --method beam).",
    )

    # Device
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )

    # Output
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed inference statistics.",
    )

    return parser


def resolve_checkpoint(checkpoint_arg: str | None) -> Path:
    """Resolves checkpoint path, falling back to the default best_model.pt."""
    if checkpoint_arg:
        return Path(checkpoint_arg).resolve()

    default = Path("E:/LipRead_Data/checkpoints/best_model.pt")
    if default.exists():
        return default

    # Try latest_model.pt as secondary fallback
    latest = Path("E:/LipRead_Data/checkpoints/latest_model.pt")
    if latest.exists():
        return latest

    raise FileNotFoundError(
        "No checkpoint found. Provide --checkpoint or train the model first.\n"
        "  Default locations checked:\n"
        f"    {default}\n"
        f"    {latest}"
    )


def run_cli(args: argparse.Namespace) -> int:
    """Executes inference from parsed CLI arguments."""
    # Resolve device
    if args.no_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Late import to keep startup fast
    from src.inference.engine import InferenceEngine

    # --- Resolve checkpoint ---
    try:
        ckpt_path = resolve_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("LipRead — Inference")
    print("=" * 70)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")
    print(f"  Method     : {args.method}" + (f"  (beam_width={args.beam_width})" if args.method == "beam" else ""))

    # --- Load engine ---
    try:
        engine = InferenceEngine.from_checkpoint(
            checkpoint_path=ckpt_path,
            decode_method=args.method,
            beam_width=args.beam_width,
            device=str(device),
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to load checkpoint: {exc}\n", file=sys.stderr)
        return 1

    print("-" * 70)

    # --- Run inference ---
    if args.video:
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            print(f"\n[ERROR] Video file not found: {video_path}\n", file=sys.stderr)
            return 1
        print(f"  Input      : {video_path}")
        print("-" * 70)
        result = engine.transcribe(video_path=video_path, decode_method=args.method, beam_width=args.beam_width)

    else:
        tensor_path = Path(args.tensor).resolve()
        if not tensor_path.exists():
            print(f"\n[ERROR] Tensor file not found: {tensor_path}\n", file=sys.stderr)
            return 1
        print(f"  Input      : {tensor_path}")
        print("-" * 70)
        try:
            frames = torch.load(str(tensor_path), weights_only=True)
        except Exception as exc:
            print(f"\n[ERROR] Failed to load tensor: {exc}\n", file=sys.stderr)
            return 1
        result = engine.transcribe_tensor(frames=frames, decode_method=args.method, beam_width=args.beam_width)

    # --- Print result ---
    print()
    if not result.success:
        print(f"[INFERENCE FAILED]\n  Reason: {result.error}")
        return 1

    print(f"  Transcription: '{result.text}'")
    print()

    if args.verbose:
        print(f"  Frames processed : {result.num_frames}")
        print(f"  Confidence       : {result.confidence:.4f}")
        print(f"  Elapsed          : {result.elapsed_ms:.1f} ms")

    print("=" * 70 + "\n")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
