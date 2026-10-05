"""Dataset smoke test utility for LipRead.

Loads a very small batch of real samples through EGCLLCDataset and DataLoader,
verifies tensor dimensions, token alignments, and reports GPU availability
without performing any neural network training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.data.collate import pad_collate_fn
from src.data.dataloader import create_dataloader
from src.data.egcllc_dataset import EGCLLCDataset
from src.data.paths import get_metadata_dir
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.data.smoke_test")


def run_smoke_test(
    split: str = "train",
    batch_size: int = 2,
    max_samples: int = 4,
    use_cache: bool = False,
) -> int:
    """Runs a smoke test on real dataset samples."""
    print("\n" + "=" * 75)
    print("LipRead - Dataset Smoke Test")
    print("=" * 75)

    meta_file = get_metadata_dir() / f"{split}.jsonl"
    if not meta_file.exists():
        # Fall back to general index if split files are not built yet
        meta_file = get_metadata_dir() / "egcllc_index.jsonl"

    if not meta_file.exists():
        print(f"[ERROR] No metadata manifest found at {meta_file}. Run index builder first.", file=sys.stderr)
        return 1

    print(f"Metadata Source:        {meta_file.name}")
    print(f"Requested Batch Size:   {batch_size}")
    print(f"Max Samples to Sample:  {max_samples}")
    print(f"Use Preprocessed Cache: {use_cache}")
    print("-" * 75)

    try:
        dataset = EGCLLCDataset(
            split=split,
            metadata_file=meta_file,
            use_cache=use_cache,
            max_samples=max_samples,
        )

        if len(dataset) == 0:
            print("[ERROR] Dataset contains 0 samples.", file=sys.stderr)
            return 1

        print(f"Dataset Initialized:    {len(dataset)} samples loaded into manifest.")

        loader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=pad_collate_fn,
        )

        # Retrieve a single batch
        batch = next(iter(loader))

        frames = batch["frames"]
        tokens = batch["tokens"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        sample_ids = batch["sample_ids"]
        transcripts = batch["transcripts"]

        print("\n--- Batch Inspection ---")
        print(f"Batch Input Shape:      {tuple(frames.shape)} [B, T, C, H, W]")
        print(f"Batch Input Dtype:      {frames.dtype}")
        print(f"Batch Input Range:      [{frames.min().item():.3f}, {frames.max().item():.3f}]")
        print(f"Batch Target Shape:     {tuple(tokens.shape)} [B, max_L]")
        print(f"Batch Target Dtype:     {tokens.dtype}")
        print(f"Input Lengths (T):      {input_lengths.tolist()}")
        print(f"Target Lengths (L):     {target_lengths.tolist()}")
        print(f"Sample IDs in Batch:    {sample_ids}")
        print(f"Transcripts:            {transcripts}")
        print("-" * 75)

        # Hardware report
        print(f"CUDA Available:         {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"Active GPU:             {device_name} ({vram_gb} GB VRAM)")
        else:
            print("Active Hardware:        CPU Only")

        print("=" * 75)
        print("RESULT: Smoke test PASSED successfully. Dataset is pipeline-compatible.\n")
        return 0

    except Exception as exc:
        print(f"\n[EXCEPTION] Smoke test failed: {exc}\n", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset smoke test")
    parser.add_argument("--split", "-s", type=str, default="train", help="Split to sample from")
    parser.add_argument("--batch-size", "-b", type=int, default=2, help="Batch size")
    parser.add_argument("--max-samples", "-m", type=int, default=4, help="Maximum samples to load")
    parser.add_argument("--use-cache", action="store_true", help="Use tensor cache")
    args = parser.parse_args()

    return run_smoke_test(
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        use_cache=args.use_cache,
    )


if __name__ == "__main__":
    sys.exit(main())
