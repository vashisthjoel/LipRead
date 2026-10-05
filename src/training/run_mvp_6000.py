"""Script executing the First Real MVP Training Run for LipRead.

Experiment Target:
- Dataset: EGCLLC speaker-disjoint subsets
  - Train: exactly 6,000 samples (speakers s1, s10, s11, s12, s13, s15, s16, s17, s18, s22, s26, s27, s28)
  - Val: 1,000 samples (speaker s19)
  - Test: 1,000 samples (speaker s14)
- Epochs: 6
- Seed: 42
- Hardware: NVIDIA GeForce GTX 1650 (4 GB VRAM)
- Model: LipReadModel (3D CNN + Bi-LSTM + CTC, 2,541,310 parameters)
- Mixed Precision: AMP (GradScaler)
- Batch size: 4, Gradient accumulation: 2 (effective batch size: 8)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-6)
- Checkpoint directory: E:\\LipRead_Data\\checkpoints\\mvp_6000\\
- Tracking: MLflow (Experiment: LipRead, Run: mvp_6000_samples_6_epochs)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

# Mute MLflow agent hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.data.dataloader import create_dataloader
from src.data.dataset_config import VocabularyConfig
from src.data.egcllc_dataset import EGCLLCDataset
from src.data.paths import get_metadata_dir, get_mlruns_dir
from src.genai.corrector import ContextualCorrector
from src.inference.decoder import greedy_decode
from src.models.ctc import CTCLossWrapper
from src.models.lipread_model import LipReadModel
from src.training.checkpoint import load_checkpoint
from src.training.scheduler import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.training.validation import evaluate
from src.utils.config import get_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lipread.mvp_6000")


def run_experiment() -> Dict[str, Any]:
    logger.info("=================================================================")
    logger.info("   LIPREAD — FIRST REAL MVP TRAINING RUN (6,000 SAMPLES)   ")
    logger.info("=================================================================")

    # 1. Deterministic Seed
    seed = 42
    set_seed(seed=seed, deterministic_cuda=False)

    # 2. Hardware / Device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GTX 1650 GPU is required for this experiment.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    logger.info(f"Target GPU: {gpu_name} ({total_vram:.2f} GB Total VRAM)")

    # 3. Metadata manifests verification
    metadata_dir = get_metadata_dir()
    train_meta = metadata_dir / "mvp_6000_train.jsonl"
    val_meta = metadata_dir / "mvp_val.jsonl"
    test_meta = metadata_dir / "mvp_test.jsonl"

    for p, label in [(train_meta, "Train"), (val_meta, "Val"), (test_meta, "Test")]:
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Required {label} metadata file missing or empty: {p}")

    # 4. Checkpoint directories safety check
    smoke_checkpoint_dir = Path("E:/LipRead_Data/checkpoints")
    mvp_checkpoint_dir = Path("E:/LipRead_Data/checkpoints/mvp_6000")
    mvp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    smoke_best = smoke_checkpoint_dir / "best_model.pt"
    if smoke_best.exists():
        logger.info(f"Verified existing smoke checkpoint untouched at: {smoke_best} ({smoke_best.stat().st_size:,} bytes)")
    logger.info(f"MVP experiment checkpoints will be saved exclusively to: {mvp_checkpoint_dir}")

    # 5. Datasets and Loaders
    vocab = VocabularyConfig()
    logger.info("Initializing EGCLLC datasets (use_cache=True)...")

    train_dataset = EGCLLCDataset(split="train", metadata_file=train_meta, use_cache=True)
    val_dataset = EGCLLCDataset(split="val", metadata_file=val_meta, use_cache=True)
    test_dataset = EGCLLCDataset(split="test", metadata_file=test_meta, use_cache=True)

    logger.info(f"Dataset Counts -> Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
    assert len(train_dataset) == 6000, f"Expected exactly 6,000 train samples, got {len(train_dataset)}"
    assert len(val_dataset) == 1000, f"Expected 1,000 val samples, got {len(val_dataset)}"
    assert len(test_dataset) == 1000, f"Expected 1,000 test samples, got {len(test_dataset)}"

    batch_size = 4
    accum_steps = 2
    num_workers = 0  # safest on Windows

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 6. Model Construction (Existing Architecture)
    cfg = get_config().to_dict()
    model_cfg = cfg.get("model", {})

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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Architecture: 3D CNN + Bi-LSTM + CTC ({total_params:,} total parameters, {trainable_params:,} trainable)")

    # 7. Optimizer & Scheduler
    epochs = 6
    lr = 1e-4
    weight_decay = 1e-4

    optimizer = build_optimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_type="AdamW",
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="plateau",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # 8. Trainer Setup
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        vocab=vocab,
        device=device,
        use_amp=True,
        gradient_accumulation_steps=accum_steps,
        gradient_clip_norm=1.0,
        checkpoint_dir=mvp_checkpoint_dir,
        tracking_uri=str(get_mlruns_dir().as_uri()),
        experiment_name="LipRead",
        run_name="mvp_6000_samples_6_epochs",
        config_dict=cfg,
        seed=seed,
    )

    # 9. Execute 6 Epochs Training Loop
    start_train_time = time.perf_counter()
    train_history = trainer.fit(epochs=epochs)
    total_train_duration = time.perf_counter() - start_train_time

    logger.info("=================================================================")
    logger.info("                 6-EPOCH TRAINING COMPLETED                      ")
    logger.info("=================================================================")
    logger.info(f"Total Training Wall Time: {total_train_duration:.2f}s ({total_train_duration/60:.2f} min)")
    logger.info(f"Best Validation Loss: {trainer.best_val_loss:.4f}")

    best_checkpoint_path = mvp_checkpoint_dir / "best_model.pt"
    latest_checkpoint_path = mvp_checkpoint_dir / "latest_model.pt"

    # 10. Held-Out Speaker-Disjoint Test Evaluation
    logger.info("\n--- Loading Best Validation Checkpoint for Final Test Evaluation ---")
    load_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=model,
        device=device,
    )

    ctc_loss_fn = CTCLossWrapper(blank=vocab.blank_idx, zero_infinity=True)

    test_start = time.perf_counter()
    test_result = evaluate(
        model=model,
        dataloader=test_loader,
        ctc_loss_fn=ctc_loss_fn,
        device=device,
        vocab=vocab,
    )
    test_duration = time.perf_counter() - test_start

    logger.info("=================================================================")
    logger.info("                 MVP 6-EPOCH TEST RESULTS                        ")
    logger.info("=================================================================")
    logger.info(f"Test Loss: {test_result.val_loss:.4f}")
    logger.info(f"Test CER:  {test_result.cer:.4f} ({test_result.cer*100:.2f}%)")
    logger.info(f"Test WER:  {test_result.wer:.4f} ({test_result.wer*100:.2f}%)")
    logger.info(f"Test Eval Duration: {test_duration:.2f}s ({test_result.samples_per_sec:.1f} samples/sec)")

    # Log test metrics to MLflow if run was active
    if trainer.mlflow_available and trainer.mlflow_run:
        try:
            import mlflow
            mlflow.log_metrics({
                "test_loss": test_result.val_loss,
                "test_cer": test_result.cer,
                "test_wer": test_result.wer,
                "test_duration_seconds": test_duration,
            })
            mlflow.log_params({"test_sample_count": len(test_dataset)})
        except Exception as e:
            logger.warning(f"Could not log test metrics to MLflow: {e}")

    # 11. Qualitative Inference Samples
    logger.info("\n--- Qualitative Test Sample Predictions ---")
    corrector = ContextualCorrector()

    qualitative_samples: List[Dict[str, Any]] = []
    # Pick 5 distinct samples from test set
    indices = [0, 100, 250, 500, 750]
    model.eval()

    with torch.no_grad():
        for idx in indices:
            item = test_dataset[idx]
            frames = item["frames"].unsqueeze(0).to(device)  # [1, T, C, H, W]
            with torch.amp.autocast("cuda"):
                logits = model(frames)  # [1, T, V]
                log_probs = logits.log_softmax(dim=-1)

            # Greedy CTC decode (handles blank stripping & repeat collapsing)
            raw_pred = greedy_decode(log_probs[0], vocab=vocab, blank_idx=vocab.blank_idx)

            # Optional GenAI contextual post-processing
            corr_res = corrector.correct(raw_pred)

            record = {
                "sample_id": item["sample_id"],
                "ground_truth": item["transcript"],
                "raw_prediction": raw_pred,
                "llm_correction": corr_res.corrected_transcription,
                "correction_applied": corr_res.correction_applied,
            }
            qualitative_samples.append(record)

            logger.info(f"Sample {item['sample_id']}:")
            logger.info(f"  Ground Truth:         {item['transcript']}")
            logger.info(f"  Raw Model Prediction: {raw_pred}")
            logger.info(f"  LLM Post-Processed:   {corr_res.corrected_transcription} (Applied: {corr_res.correction_applied})")

    # 12. Save Results Summary JSON
    results_summary = {
        "experiment": "mvp_6000_samples_6_epochs",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": gpu_name,
            "vram_gb": total_vram,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2),
        },
        "dataset_split": {
            "train_speakers": sorted(list({"s1", "s10", "s11", "s12", "s13", "s15", "s16", "s17", "s18", "s22", "s26", "s27", "s28"})),
            "val_speakers": ["s19"],
            "test_speakers": ["s14"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
        },
        "training_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": accum_steps,
            "effective_batch_size": batch_size * accum_steps,
            "initial_lr": lr,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "use_amp": True,
            "seed": seed,
        },
        "training_history": train_history.get("history", []),
        "best_validation": {
            "val_loss": trainer.best_val_loss,
            "best_checkpoint": str(best_checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint_path),
        },
        "test_evaluation": {
            "test_loss": test_result.val_loss,
            "test_cer": test_result.cer,
            "test_wer": test_result.wer,
            "test_duration_seconds": test_duration,
        },
        "qualitative_samples": qualitative_samples,
    }

    results_file = metadata_dir / "mvp_6000_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Complete experiment summary written to: {results_file}")

    return results_summary


if __name__ == "__main__":
    run_experiment()
