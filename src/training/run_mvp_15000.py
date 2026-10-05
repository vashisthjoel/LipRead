"""Script executing the Full MVP-2 Training Run (15,000 Samples, 10 Epochs) for LipRead.

Approved MVP-2 Specification:
- Dataset: EGCLLC speaker-disjoint subsets
  - Train: exactly 15,000 samples (16 speakers: s1, s2, s10, s11, s12, s13, s15, s16, s17, s18, s22, s24, s25, s26, s27, s28)
  - Val: exactly 2,000 samples (speakers s19, s20: 1,000 each)
  - Test: exactly 2,000 samples (speakers s14, s23: 1,000 each) - held out until after epoch 10
- Epochs: 10 (no early stopping)
- Seed: 42
- Hardware: NVIDIA GeForce GTX 1650 (4 GB VRAM)
- Model: LipReadModel (3D CNN + Bi-LSTM + CTC, 2,541,310 parameters)
- Mixed Precision: AMP (GradScaler)
- Batch size: 4, Gradient accumulation: 2 (effective batch size: 8)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-6, mode="min")
- Checkpoint directory: E:\\LipRead_Data\\checkpoints\\mvp_15000\\
- Tracking: MLflow (Experiment: LipRead, Run: mvp_15000_samples_10_epochs)
- Results: E:\\LipRead_Data\\metadata\\mvp2_results.json
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

from src.data.collate import pad_collate_fn
from src.data.dataloader import create_dataloader
from src.data.dataset_config import VocabularyConfig
from src.data.egcllc_dataset import EGCLLCDataset
from src.data.paths import get_metadata_dir, get_mlruns_dir
from src.models.ctc import CTCLossWrapper, decode_to_text
from src.models.lipread_model import LipReadModel
from src.training.checkpoint import load_checkpoint
from src.training.metrics import compute_batch_cer, compute_batch_wer, compute_cer, compute_wer
from src.training.scheduler import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import get_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lipread.mvp_15000")


def evaluate_test_set(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ctc_loss_fn: CTCLossWrapper,
    device: torch.device,
    vocab: VocabularyConfig,
) -> Dict[str, Any]:
    """Runs a complete greedy evaluation pass on held-out test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_samples = 0

    all_records: List[Dict[str, Any]] = []
    all_references: List[str] = []
    all_hypotheses: List[str] = []
    exact_matches = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
            video = batch["frames"].to(device, non_blocking=True)
            targets = batch["tokens"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            transcripts = batch["transcripts"]
            sample_ids = batch.get("sample_ids", [""] * len(transcripts))

            b_size = video.size(0)
            total_samples += b_size

            with torch.amp.autocast("cuda"):
                logits = model(video)
                ctc_logits = model.to_ctc_layout(logits)
                loss = ctc_loss_fn(ctc_logits, targets, input_lengths, target_lengths)

            total_loss += loss.item()
            num_batches += 1

            hypotheses = decode_to_text(logits, blank_idx=vocab.blank_idx, vocab=vocab)
            references = [t.strip() for t in transcripts]

            all_references.extend(references)
            all_hypotheses.extend(hypotheses)

            for sid, ref, hyp in zip(sample_ids, references, hypotheses):
                c_err = compute_cer(ref, hyp)
                w_err = compute_wer(ref, hyp)
                is_exact = int(ref.strip().lower() == hyp.strip().lower())
                exact_matches += is_exact
                spk = sid.split("/")[0] if "/" in sid else "unknown"
                all_records.append({
                    "sample_id": sid,
                    "speaker_id": spk,
                    "reference": ref,
                    "prediction": hyp,
                    "cer": c_err,
                    "wer": w_err,
                    "exact_match": is_exact,
                })

    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    samples_per_sec = total_samples / max(elapsed, 1e-4)

    cer = compute_batch_cer(all_references, all_hypotheses)
    wer = compute_batch_wer(all_references, all_hypotheses)
    exact_acc = exact_matches / max(total_samples, 1)

    return {
        "test_loss": avg_loss,
        "test_cer": cer,
        "test_wer": wer,
        "exact_sentence_matches": exact_matches,
        "total_test_samples": total_samples,
        "exact_sentence_accuracy": exact_acc,
        "elapsed_sec": elapsed,
        "samples_per_sec": samples_per_sec,
        "sample_records": all_records,
    }


def run_experiment() -> Dict[str, Any]:
    logger.info("=================================================================")
    logger.info("   LIPREAD — FULL MVP-2 TRAINING RUN (15,000 SAMPLES, 10 EPOCHS)   ")
    logger.info("=================================================================")

    # 1. Deterministic Seed
    seed = 42
    set_seed(seed=seed, deterministic_cuda=False)

    # 2. Hardware / Device Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GTX 1650 GPU is required for this experiment.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    logger.info(f"Target GPU: {gpu_name} ({total_vram:.2f} GB Total VRAM)")

    # 3. Pre-flight Check: Metadata manifests verification
    metadata_dir = get_metadata_dir()
    train_meta = metadata_dir / "mvp2_train.jsonl"
    val_meta = metadata_dir / "mvp2_val.jsonl"
    test_meta = metadata_dir / "mvp2_test.jsonl"

    for p, label in [(train_meta, "Train"), (val_meta, "Val"), (test_meta, "Test")]:
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Required {label} metadata file missing or empty: {p}")

    # 4. Checkpoint Safety Check
    mvp1_checkpoint_dir = Path("E:/LipRead_Data/checkpoints/mvp_6000")
    mvp1_best = mvp1_checkpoint_dir / "best_model.pt"
    if not mvp1_best.exists():
        raise FileNotFoundError(f"MVP-1 checkpoint missing: {mvp1_best}")
    logger.info(f"Verified MVP-1 checkpoint untouched at: {mvp1_best} ({mvp1_best.stat().st_size:,} bytes)")

    mvp2_checkpoint_dir = Path("E:/LipRead_Data/checkpoints/mvp_15000")
    mvp2_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"MVP-2 checkpoints will be saved exclusively to: {mvp2_checkpoint_dir}")

    # 5. Datasets and Loaders
    vocab = VocabularyConfig()
    logger.info("Initializing MVP-2 EGCLLC datasets (use_cache=True)...")

    train_dataset = EGCLLCDataset(split="train", metadata_file=train_meta, use_cache=True)
    val_dataset = EGCLLCDataset(split="val", metadata_file=val_meta, use_cache=True)
    test_dataset = EGCLLCDataset(split="test", metadata_file=test_meta, use_cache=True)

    logger.info(f"Dataset Counts -> Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
    assert len(train_dataset) == 15000, f"Expected exactly 15,000 train samples, got {len(train_dataset)}"
    assert len(val_dataset) == 2000, f"Expected exactly 2,000 val samples, got {len(val_dataset)}"
    assert len(test_dataset) == 2000, f"Expected exactly 2,000 test samples, got {len(test_dataset)}"

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

    # 6. Model Construction (Exact same architecture as MVP-1)
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
    epochs = 10
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
        checkpoint_dir=mvp2_checkpoint_dir,
        tracking_uri=str(get_mlruns_dir().as_uri()),
        experiment_name="LipRead",
        run_name="mvp_15000_samples_10_epochs",
        config_dict=cfg,
        seed=seed,
    )

    # 9. Execute 10 Epochs Training Loop
    logger.info(f"Beginning MVP-2 training run ({epochs} epochs, no early stopping)...")
    start_train_time = time.perf_counter()
    train_history = trainer.fit(epochs=epochs)
    total_train_duration = time.perf_counter() - start_train_time

    logger.info("=================================================================")
    logger.info("                10-EPOCH TRAINING COMPLETED                      ")
    logger.info("=================================================================")
    logger.info(f"Total Training Wall Time: {total_train_duration:.2f}s ({total_train_duration/60:.2f} min)")
    logger.info(f"Best Validation Loss: {trainer.best_val_loss:.4f}")

    best_checkpoint_path = mvp2_checkpoint_dir / "best_model.pt"
    latest_checkpoint_path = mvp2_checkpoint_dir / "latest_model.pt"

    # Find best epoch from history
    best_epoch_info = min(train_history.get("history", []), key=lambda h: h["val_loss"], default={})
    best_epoch = best_epoch_info.get("epoch", -1)
    best_val_cer = best_epoch_info.get("cer", 1.0)
    best_val_wer = best_epoch_info.get("wer", 1.0)

    # 10. Held-Out Speaker-Disjoint Test Evaluation (s14, s23)
    logger.info("\n--- Loading Best Validation Checkpoint for Final Test Evaluation ---")
    load_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=model,
        device=device,
    )

    ctc_loss_fn = CTCLossWrapper(blank=vocab.blank_idx, zero_infinity=True)

    logger.info("Evaluating 2,000 held-out test samples (s14, s23) with greedy decoding...")
    test_res = evaluate_test_set(
        model=model,
        dataloader=test_loader,
        ctc_loss_fn=ctc_loss_fn,
        device=device,
        vocab=vocab,
    )

    logger.info("=================================================================")
    logger.info("                 MVP-2 10-EPOCH TEST RESULTS                     ")
    logger.info("=================================================================")
    logger.info(f"Test Loss:             {test_res['test_loss']:.4f}")
    logger.info(f"Test CER:              {test_res['test_cer']:.4f} ({test_res['test_cer']*100:.2f}%)")
    logger.info(f"Test WER:              {test_res['test_wer']:.4f} ({test_res['test_wer']*100:.2f}%)")
    logger.info(f"Exact Sentence Match:  {test_res['exact_sentence_matches']} / {test_res['total_test_samples']} ({test_res['exact_sentence_accuracy']*100:.2f}%)")
    logger.info(f"Test Eval Duration:    {test_res['elapsed_sec']:.2f}s ({test_res['samples_per_sec']:.1f} samples/sec)")

    # Log test metrics to MLflow run
    mlflow_run_id = train_history.get("mlflow_run_id")
    if trainer.mlflow_available and mlflow_run_id and mlflow_run_id != "none":
        try:
            import mlflow
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_metrics({
                    "test_loss": test_res["test_loss"],
                    "test_cer": test_res["test_cer"],
                    "test_wer": test_res["test_wer"],
                    "test_exact_sentence_accuracy": test_res["exact_sentence_accuracy"],
                    "test_duration_seconds": test_res["elapsed_sec"],
                    "total_training_duration_seconds": total_train_duration,
                })
                mlflow.log_params({"test_sample_count": len(test_dataset)})
        except Exception as e:
            logger.warning(f"Could not log test metrics to MLflow: {e}")

    # 11. Extract Qualitative Samples (at least 10 deterministic examples)
    # Pick 15 evenly-spaced samples across the 2,000 test set
    sample_records = test_res.pop("sample_records")
    indices = [int(i * (len(sample_records) - 1) / 14) for i in range(15)]
    qualitative_examples = [sample_records[idx] for idx in indices]

    logger.info("\n--- Qualitative Test Examples (Greedy CTC) ---")
    for i, ex in enumerate(qualitative_examples, 1):
        logger.info(
            f"[{i:02d}] Sample: {ex['sample_id']} | CER: {ex['cer']*100:.1f}% | WER: {ex['wer']*100:.1f}%\n"
            f"     Ref:  '{ex['reference']}'\n"
            f"     Pred: '{ex['prediction']}'"
        )

    # 12. Save Results Summary JSON
    train_speakers_list = sorted(list({"s1", "s2", "s10", "s11", "s12", "s13", "s15", "s16", "s17", "s18", "s22", "s24", "s25", "s26", "s27", "s28"}))
    results_summary = {
        "experiment": "mvp_15000_samples_10_epochs",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": gpu_name,
            "vram_gb": total_vram,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2),
        },
        "dataset_split": {
            "train_speakers": train_speakers_list,
            "val_speakers": ["s19", "s20"],
            "test_speakers": ["s14", "s23"],
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
            "best_epoch": best_epoch,
            "best_val_loss": trainer.best_val_loss,
            "best_val_cer": best_val_cer,
            "best_val_wer": best_val_wer,
            "best_checkpoint": str(best_checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint_path),
        },
        "test_evaluation": test_res,
        "qualitative_examples": qualitative_examples,
        "timing": {
            "total_train_duration_sec": total_train_duration,
            "total_test_duration_sec": test_res["elapsed_sec"],
        },
        "mlflow": {
            "experiment": "LipRead",
            "run_id": mlflow_run_id,
        },
    }

    results_file = metadata_dir / "mvp2_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Complete MVP-2 experiment summary written to: {results_file}")

    return results_summary


if __name__ == "__main__":
    run_experiment()
