"""CLI entry point for training LipRead on EGCLLC.

Usage:
    python -m src.training.train --epochs 5 --batch-size 2
    python -m src.training.train --epochs 1 --max-train-batches 100 --max-val-batches 20
    python -m src.training.train --resume E:\\LipRead_Data\\checkpoints\\latest_model.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# Ensure local warnings from mlflow are muted
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from src.data.dataloader import create_dataloader
from src.data.dataset_config import VocabularyConfig
from src.data.egcllc_dataset import EGCLLCDataset
from src.data.paths import get_checkpoint_dir, get_metadata_dir, get_mlruns_dir
from src.models.lipread_model import LipReadModel
from src.training.checkpoint import load_checkpoint
from src.training.scheduler import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import get_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lipread.train")


def validate_environment_preflight(
    metadata_dir: Path,
    vocab: VocabularyConfig,
    device: torch.device,
) -> None:
    """Performs rigorous pre-flight validation before launching training.

    Ensures manifests, files, vocabulary, model, and device forward passes work.
    Raises RuntimeError if any check fails.
    """
    logger.info("--- Starting Pre-Flight Training Environment Verification ---")

    # 1. Manifests
    for split_name in ("train", "val", "test"):
        manifest_path = metadata_dir / f"{split_name}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Pre-flight failed: Split manifest missing: {manifest_path}. "
                "Ensure Stage 3 EGCLLC indexing has run."
            )
        size = manifest_path.stat().st_size
        if size == 0:
            raise ValueError(f"Pre-flight failed: Split manifest {manifest_path} is empty.")
        logger.info(f"Verified {split_name} manifest: {manifest_path.name} ({size:,} bytes)")

    # 2. Vocabulary check
    if vocab.vocab_size <= 2:
        raise ValueError(f"Pre-flight failed: Vocabulary size is too small ({vocab.vocab_size}).")
    logger.info(f"Verified Vocabulary: {vocab.vocab_size} classes (blank=0, unk={vocab.unk_idx})")

    # 3. CUDA check
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        gpu_name = torch.cuda.get_device_name(device)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        logger.info(f"Verified CUDA GPU: {gpu_name} ({total_vram:.2f} GB total VRAM)")
    else:
        logger.info("Verified Device: CPU (AMP disabled)")

    logger.info("--- Pre-Flight Verification PASSED Successfully ---\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LipRead - Training Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default from config: 5)")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size (default: 2)")
    parser.add_argument("--accum-steps", type=int, default=None, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--device", type=str, default=None, help="Target device: 'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit training batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches per epoch")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--use-cache", action="store_true", help="Enable preprocessed tensor disk caching")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config().to_dict()
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # Resolve settings with CLI overrides
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    set_seed(seed=seed, deterministic_cuda=False)

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 5)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 2)
    accum_steps = args.accum_steps if args.accum_steps is not None else train_cfg.get("gradient_accumulation_steps", 2)
    lr = args.lr if args.lr is not None else train_cfg.get("initial_learning_rate", 1e-4)
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 1e-4)
    use_amp = not args.no_amp if args.no_amp else train_cfg.get("mixed_precision", True)

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    metadata_dir = get_metadata_dir()
    vocab = VocabularyConfig()

    # 1. Environment validation
    validate_environment_preflight(metadata_dir, vocab, device)

    # 2. Datasets and DataLoaders
    logger.info("Initializing lazy EGCLLC train and validation datasets...")
    train_dataset = EGCLLCDataset(split="train", metadata_file=metadata_dir / "train.jsonl", use_cache=args.use_cache, filter_extracted=True)
    val_dataset = EGCLLCDataset(split="val", metadata_file=metadata_dir / "val.jsonl", use_cache=args.use_cache, filter_extracted=True)

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        f"Datasets loaded: Train samples={len(train_dataset):,}, Val samples={len(val_dataset):,}. "
        f"Batch size={batch_size}, Grad accum={accum_steps} (Effective batch={batch_size * accum_steps})"
    )

    # 3. Model construction
    model = LipReadModel(
        vocab_size=vocab.vocab_size,
        cnn_channels=model_cfg.get("cnn_channels", [32, 64, 96]),
        lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
        lstm_layers=model_cfg.get("lstm_layers", 2),
        bidirectional=model_cfg.get("bidirectional", True),
        dropout=model_cfg.get("dropout", 0.2),
        ctc_blank_index=vocab.blank_idx,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = build_optimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_type=train_cfg.get("optimizer", "AdamW"),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="plateau",
        mode="min",
        factor=0.5,
        patience=train_cfg.get("early_stopping_patience", 3),
        min_lr=1e-6,
    )

    # 5. Checkpoint Resume if specified
    checkpoint_dir = get_checkpoint_dir()
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        vocab=vocab,
        device=device,
        use_amp=use_amp,
        gradient_accumulation_steps=accum_steps,
        gradient_clip_norm=train_cfg.get("gradient_clip_norm", 1.0),
        checkpoint_dir=checkpoint_dir,
        tracking_uri=str(get_mlruns_dir().as_uri()),
        experiment_name=cfg.get("mlflow", {}).get("experiment_name", "LipRead"),
        config_dict=cfg,
        seed=seed,
    )

    # 7. Execute training
    results = trainer.fit(
        epochs=epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    logger.info("Training session finished.")
    logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    if results["best_checkpoint_path"]:
        logger.info(f"Best Checkpoint: {results['best_checkpoint_path']}")


if __name__ == "__main__":
    main()
