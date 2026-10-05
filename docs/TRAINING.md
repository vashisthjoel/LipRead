# LipRead — Training Pipeline Documentation

This document describes the environment, dataset setup, speaker-disjoint partitioning, training mechanics, experiment tracking, and evaluation procedures for the **LipRead** model.

---

## 1. Hardware & Software Environment

- **Target Compute:** NVIDIA GeForce GTX 1650 (4.00 GB VRAM)
- **Host System:** Windows 11 (tested on Python 3.12.4, CUDA 12.6, PyTorch 2.14.1)
- **Precision:** Automated Mixed Precision (`torch.amp.autocast("cuda")` with `torch.amp.GradScaler`)
- **Batching Strategy:**
  - Micro-batch size: 4
  - Gradient accumulation steps: 2
  - Effective batch size: 8

---

## 2. Dataset & Speaker Partitioning

LipRead uses sentence-level lip-reading data derived from the **EGCLLC** (Enhanced GRID Corpus with Lip Landmark Coordinates) dataset.

### Strict Speaker-Disjoint Splitting
To evaluate genuine generalization across unseen speakers, the train, validation, and test splits are strictly speaker-disjoint with **zero speaker overlap**.

| Split | Speakers | Sample Count | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | `s1, s2, s10, s11, s12, s13, s15, s16, s17, s18, s22, s24, s25, s26, s27, s28` (16 speakers) | 15,000 | Model parameter optimization |
| **Validation** | `s19, s20` (2 speakers) | 2,000 | Model checkpoint selection & hyperparameter tuning |
| **Test** | `s14, s23` (2 speakers) | 2,000 | Strictly held-out final generalization benchmark |

*Note: In the earlier MVP-1 baseline experiment, 6,000 training samples and 1,000 validation samples were utilized.*

---

## 3. Preprocessing & Dynamic Disk Caching

During dataset iteration, video sequences are preprocessed as follows:
1. Load 75 JPEG images per utterance.
2. Crop mouth region of interest (ROI) via MediaPipe 478-landmark FaceMesh.
3. Convert to grayscale, resize to $88 \times 88$, and normalize pixels to $[0.0, 1.0]$.
4. Format into a 4D tensor `[75, 1, 88, 88]`.
5. **Disk Caching:** When `use_cache=True` is configured, preprocessed tensors are serialized to disk as `.pt` files. Subsequent epochs read the preprocessed tensors directly, bypassing JPEG decoding and facial landmark detection entirely.

---

## 4. Hyperparameters & Optimizer Configuration

The training configuration is defined in [`configs/config.yaml`](../configs/config.yaml):

```yaml
training:
  epochs: 10
  batch_size: 4
  gradient_accumulation_steps: 2
  effective_batch_size: 8
  learning_rate: 0.0001
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  seed: 42

optimizer:
  type: "AdamW"
  lr: 1.0e-4
  weight_decay: 1.0e-4

scheduler:
  type: "plateau"
  mode: "min"
  factor: 0.5
  patience: 2
  min_lr: 1.0e-6
```

---

## 5. Loss Function & Evaluation Metrics

- **Loss:** Connectionist Temporal Classification (CTC) Loss with zero-infinity enabled (`nn.CTCLoss(blank=0, zero_infinity=True)`).
- **Character Error Rate (CER):** Levenshtein edit distance between ground truth transcript and greedy CTC decoded transcript, normalized by reference character count:
  $$\text{CER} = \frac{S + D + I}{N_{\text{chars}}}$$
- **Word Error Rate (WER):** Levenshtein distance on space-delimited words:
  $$\text{WER} = \frac{S + D + I}{N_{\text{words}}}$$

---

## 6. Experiment Tracking (MLflow)

All training runs are logged to local MLflow file store at `mlruns/` (or `$LIPREAD_DATA_ROOT/mlruns`):
- Hyperparameters: batch size, learning rate, gradient clipping norm, model parameters, seed.
- Epoch metrics: `train_loss`, `val_loss`, `cer`, `wer`, `learning_rate`, `epoch_time_sec`, `vram_peak_mb`.
- Best validation loss checkpoint tracking.

To launch the MLflow tracking UI:
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

---

## 7. Execution Commands

### Running Training via CLI
```bash
python -m src.training.train --config configs/config.yaml
```

---

## 8. Current Experiments Status

### MVP-1 Baseline (Completed)
- **Dataset:** 6,000 train / 1,000 val / 1,000 test.
- **Duration:** 6 epochs.
- **Outcome:** Successfully converged and completed full test evaluation. Test Loss: 1.2611, CER: 55.08%, WER: 97.48%.

### MVP-2 Scale Run (🟡 In Progress)
- **Dataset:** 15,000 train / 2,000 val / 2,000 test.
- **Epoch 1 Execution:** All 15,000 training samples (3,750 batches) trained successfully with zero numerical errors, NaN, or CUDA exceptions.
- **Validation Interruption & Windows Cache Limitation:** During validation on sample 1,168 of 2,000, the process halted with Windows `[WinError 1450] Insufficient system resources exist to complete the requested service`.
  - *Cause:* Having 17,277 `.pt` files in a single flat directory on an external USB NTFS drive caused repeated `os.stat` filesystem calls inside the dataloader fetcher to exhaust the Windows kernel non-paged pool.
  - *Planned Fix:* In-memory indexing of cached filenames during dataset initialization instead of issuing repeated disk stat calls.
  - *Checkpoint Status:* Weights were not serialized prior to the OS crash. All 15,000 training samples remain 100% preprocessed and cached on disk for rapid re-execution.
