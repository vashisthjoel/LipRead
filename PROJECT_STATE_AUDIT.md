# LipRead — Post-Shutdown Project State Audit

**Audit Timestamp:** 2026-10-04 22:35:00 IST  
**Target Project:** `D:\Lipreading end to end detection`  
**Dataset Root:** `E:\LipRead_Data`  
**Execution Mode:** Read-Only Audit & Inspection (Zero Mutations Applied)

---

## Executive Summary

Following an unexpected system shutdown during dataset extraction, this audit was conducted to establish the exact post-crash status across all project subsystems: active processes, dataset archives, extracted files, source code modules, PyTorch checkpoints, MLflow experiment records, test suite results, hardware environment, and Git repository state.

**Key Findings:**
1. **No background extraction or training processes are running.** The previous tar extraction halted abruptly when the system went down at approximately 22:13 IST.
2. **Dataset Extraction is Incomplete (Partial State):**
   - Alignments (`GRID_alignments`) and coordinates (`lip_coordinates`) are 100% extracted (33/33 speakers).
   - Video frames (`lip_images`): 20 speakers are fully extracted, 1 speaker (`s29`) was cut off mid-extraction (752/1000 utterances, last folder `pwwozn` is empty), and 12 speakers were not reached. Overall speaker extraction progress in `lip_images` is **60.6% complete** (20/33 complete, 1 partial, 12 unextracted).
3. **Core ML & Inference Architecture is 100% Implemented & Passing:**
   - Preprocessing (OpenCV, MediaPipe FaceMesh, 88×88 lip ROI cropping, sequence padding/truncation to 75 frames).
   - Dataset adapter, JSONL indexers, collator, PyTorch DataLoaders.
   - 3D CNN (3-stage spatiotemporal feature extractor) + Bi-LSTM (2-layer temporal encoder) + Linear projection (30 vocabulary tokens).
   - PyTorch CTC loss wrapper, Greedy CTC decoder, and Prefix Beam Search CTC decoder.
   - Full Trainer with mixed-precision (AMP), gradient accumulation, gradient clipping, ReduceLROnPlateau scheduler, checkpointing, and MLflow logging.
   - End-to-end InferenceEngine and CLI.
   - **PyTest Suite:** **72 / 72 unit tests passed (100%)** in 68.49 seconds.
4. **Current Checkpoints are Smoke-Test Sanity Artifacts, NOT Converged Models:**
   - Checkpoints (`best_model.pt`, `latest_model.pt`) represent a 100-batch sanity check (`global_step=100`, 200 samples).
   - CER and WER are both **1.0 (100%)**, as expected for random weights trained for only 100 steps.
5. **No Local Git Repository Exists:** `D:\Lipreading end to end detection` does not have a `.git` folder initialized.
6. **Application Layers Are Not Implemented:** FastAPI (`api/`), Streamlit UI (`app/`), LangChain/LLM post-processor, Dockerfiles, and GitHub Actions workflows (.gitkeep only) are completely unbuilt.

---

## 1. Dataset & Extraction State

### Process Inspection
- **Active Extraction Processes:** **NONE**
- Verified using OS process manager queries for `tar`, `7z`, `python`, and `powershell`.
- No orphan processes are executing. No extraction process was restarted during this audit.

### Filesystem Inspection (`E:\LipRead_Data\raw\EGCLLC`)
The GRID corpus within EGCLLC contains 33 valid speakers (`s1` through `s20` and `s22` through `s34`; note that speaker `s21` does not exist in the GRID corpus).

| Dataset Component | Target Path | Extracted Speakers | State | Evidence / Details |
|---|---|---|---|---|
| **GRID Alignments** | `E:\LipRead_Data\raw\EGCLLC\GRID_alignments` | 33 / 33 | **COMPLETE** | All 33 speaker directories exist (`s1`–`s20`, `s22`–`s34`). Modified 2023-12-09. |
| **Lip Coordinates** | `E:\LipRead_Data\raw\EGCLLC\lip_coordinates` | 33 / 33 | **COMPLETE** | All 33 speaker directories exist (`s1`–`s20`, `s22`–`s34`). Modified 2023-12-09. |
| **Lip Images (Frames)** | `E:\LipRead_Data\raw\EGCLLC\lip_images` | 20 Complete, 1 Partial, 12 Missing | **PARTIAL** | Sequential tar extraction proceeded alphabetically (`s1`, `s10`..`s19`, `s2`, `s20`, `s22`..`s29`). Halted during `s29`. |

### Detailed Speaker Status for `lip_images`:
- **COMPLETE (20 Speakers):**  
  `s1` (989 utterances), `s10`, `s11`, `s12`, `s13`, `s14`, `s15`, `s16`, `s17`, `s18`, `s19`, `s2`, `s20`, `s22`, `s23`, `s24`, `s25`, `s26`, `s27`, `s28` (1,000 utterances).
- **PARTIAL (1 Speaker):**  
  `s29`: Contains 752 utterances extracted out of ~1,000.  
  - Last complete directory: `s29\video\mpg_6000\pwwo3a` (75 frames).
  - Interrupted directory: `s29\video\mpg_6000\pwwozn` (timestamp: 2026-10-04 22:13, contains 0 files).
- **NOT EXTRACTED / MISSING (12 Speakers):**  
  `s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `s30`, `s31`, `s32`, `s33`, `s34`. Directories do not exist under `lip_images`.
- **Raw Tar Archive:** `E:\LipRead_Data\raw\EGCLLC_repo\lip_images.tar` exists intact (14,338,662,400 bytes / ~13.35 GiB).

### Split vs. Extraction Alignment:
According to `E:\LipRead_Data\metadata\split_summary.json`:
- **Train split (23 speakers, 22,834 samples):** 13 extracted (`s1`, `s10`-`s13`, `s15`-`s18`, `s22`, `s26`-`s28`), 10 unextracted (`s3`, `s5`-`s9`, `s30`-`s33`).
- **Val split (5 speakers, 5,000 samples):** 4 extracted (`s19`, `s20`, `s23`, `s24`), 1 partial (`s29`).
- **Test split (5 speakers, 4,993 samples):** 3 extracted (`s14`, `s2`, `s25`), 2 unextracted (`s4`, `s34`).

---

## 2. Project Source Code Inspection

Inspection of `D:\Lipreading end to end detection\src\` revealed a modular, cleanly written codebase:

| Stage / Subsystem | Source Location | Implementation Status | Features Verified |
|---|---|---|---|
| **Video Ingestion** | `src/preprocessing/video_reader.py` | **IMPLEMENTED** | OpenCV frame decoding, metadata extraction (FPS, frame count, resolution), synthetic video generation for testing. |
| **Landmark Tracking** | `src/preprocessing/landmarks.py` | **IMPLEMENTED** | MediaPipe FaceLandmarker task (478 landmarks), fallback heuristic handling for missing faces. Model asset present: `weights/face_landmarker.task`. |
| **ROI Extraction** | `src/preprocessing/lip_roi.py` | **IMPLEMENTED** | Dynamic lip bounding box calculation, square expansion, boundary clamping, 88×88 resizing, grayscale conversion, [0, 1] normalization. |
| **Sequence Building** | `src/preprocessing/sequence.py` | **IMPLEMENTED** | Temporal normalization to fixed length $T=75$, center truncation for long sequences, edge/zero padding for short sequences, tensor layout conversion `[T, C, H, W]`. |
| **Preprocessing Pipeline & CLI** | `src/preprocessing/pipeline.py`, `cli.py` | **IMPLEMENTED** | Integrated multi-frame pipeline, CLI interface supporting video file inputs with output tensor saving. |
| **Dataset Configuration** | `src/data/dataset_config.py` | **IMPLEMENTED** | Character vocabulary config (30 classes: blank at 0, space at 1, 'a'-'z', apostrophe). |
| **Dataset Adapter & Indexer** | `src/data/egcllc_adapter.py`, `egcllc_cli.py` | **IMPLEMENTED** | Parsing GRID `.align` word/phoneme timing files, tokenization, speaker-aware split generation (`train.jsonl`, `val.jsonl`, `test.jsonl`). |
| **Dataset & DataLoader** | `src/data/egcllc_dataset.py`, `collate.py`, `dataloader.py` | **IMPLEMENTED** | PyTorch Dataset with lazy loading, optional memory caching, `filter_extracted` safety flag, dynamic sequence padding collation (`pad_collate_fn`), multi-worker DataLoader. |
| **Spatiotemporal 3D CNN** | `src/models/cnn3d.py` | **IMPLEMENTED** | 3-stage spatiotemporal Conv3d blocks with BatchNorm3d, ReLU, and temporal-preserving MaxPool3d `(1, 2, 2)`. Spatial feature dimension: 96. |
| **Bidirectional LSTM** | `src/models/bilstm.py` | **IMPLEMENTED** | 2-layer Bi-LSTM (hidden size: 256 per direction, combined: 512) with dropout (0.2). |
| **End-to-End Model** | `src/models/lipread_model.py` | **IMPLEMENTED** | Unified network combining 3D CNN + Bi-LSTM + Linear projection `(512 -> 30)`. Includes batch-first to CTC layout `[T, B, V]` helpers. |
| **CTC Loss** | `src/models/ctc.py` | **IMPLEMENTED** | `torch.nn.CTCLoss` wrapper with `zero_infinity=True`, sequence length validation, loss computation. |
| **Training Pipeline** | `src/training/trainer.py`, `train.py`, `smoke_train.py` | **IMPLEMENTED** | Mixed precision training (`torch.cuda.amp.GradScaler`), gradient accumulation, gradient norm clipping (1.0), validation loop, early stopping, checkpointing. |
| **Optimization & Schedulers** | `src/training/scheduler.py` | **IMPLEMENTED** | AdamW / Adam / SGD builders, ReduceLROnPlateau and CosineAnnealing schedulers. |
| **Evaluation Metrics** | `src/training/metrics.py` | **IMPLEMENTED** | Dynamic programming Levenshtein distance, Character Error Rate (CER), Word Error Rate (WER). |
| **Checkpoint Management** | `src/training/checkpoint.py` | **IMPLEMENTED** | Atomic checkpoint saving, best model tracking based on validation loss, resume state loading. |
| **MLflow Tracking** | `src/training/trainer.py` | **IMPLEMENTED** | Experiment creation, hyperparameter logging, step/epoch metrics logging (train loss, val loss, CER, WER, VRAM usage). |
| **CTC Decoders** | `src/inference/decoder.py` | **IMPLEMENTED** | Greedy best-path decoder and Prefix Beam Search decoder with blank collapsing and repeated token deduplication. |
| **Inference Engine & CLI** | `src/inference/engine.py`, `cli.py` | **IMPLEMENTED** | `InferenceEngine` supporting raw video or preprocessed tensor inputs, checkpoint loading, confidence score generation, CLI tool. |
| **FastAPI REST Service** | `api/` | **NOT IMPLEMENTED** | Only empty package `__init__.py` exists. No endpoints or server logic implemented. |
| **Streamlit Web UI** | `app/` | **NOT IMPLEMENTED** | Only empty package `__init__.py` exists. No dashboard UI implemented. |
| **LangChain / LLM Post-Processing** | N/A | **NOT IMPLEMENTED** | Listed in requirements and README, but no implementation code exists in `src/`. |
| **Docker Containerization** | N/A | **NOT IMPLEMENTED** | No `Dockerfile` or `docker-compose.yml` exists. |
| **GitHub Actions CI/CD** | `.github/workflows/` | **NOT IMPLEMENTED** | Contains only `.gitkeep`. No automated workflow definitions exist. |
| **Unit Test Suite** | `tests/` | **IMPLEMENTED** | 6 test modules covering environment, preprocessing, dataset, models, training, and inference. |

---

## 3. Checkpoint Inspection

Located at: `E:\LipRead_Data\checkpoints\`

| Checkpoint File | Size | Last Modified | State Validity | Internal Contents |
|---|---|---|---|---|
| `best_model.pt` | 30,533,367 bytes (29.12 MB) | 2026-10-04 17:01:36 | **VALID** | `epoch: 1`, `global_step: 100`, `train_loss: 4.4036`, `val_loss: 2.9172`, `cer: 1.0`, `wer: 1.0`, `model_state_dict` (36 tensors), `optimizer_state_dict` (AdamW), `scheduler_state_dict`, `scaler_state_dict` (AMP), `config`, `random_seed: 42`. |
| `latest_model.pt` | 30,533,583 bytes (29.12 MB) | 2026-10-04 17:01:36 | **VALID** | Exact replica of step 100 state. |

**Technical Assessment:**  
The checkpoint format is structurally complete, robust, and correctly serializes all model, optimizer, scheduler, and AMP scaler parameters. However, it is **strictly an artifact of a smoke-training sanity run** (100 steps). It cannot be used for meaningful speech recognition until trained on full epochs.

---

## 4. MLflow Experiment & Run Records

Located at: `E:\LipRead_Data\mlruns\`  
Experiment Name: `LipRead` (Experiment ID: `342918310669993023`)

Three individual execution runs were recorded:

| Run ID | Run Name | Status | Duration | Train Loss | Val Loss | CER | WER | Peak VRAM | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `a9fb45def3f7413fabaae0b792cd3650` | `run_epochs_1_batch_2` | **FINISHED** (3) | 190.09s | 5.5933 | 3.0599 | 1.0 (100%) | 1.0 (100%) | 1,311.25 MB | Initial smoke verification run. |
| `0340ac3dfdc44f6c969c7ce02d31366a` | `run_epochs_1_batch_2` | **FAILED** (1) | Null | N/A | N/A | N/A | N/A | N/A | Interrupted before any metric logging occurred. |
| `772029c976c74934856a0bc78a089c93` | `run_epochs_1_batch_2` | **FINISHED** (3) | 769.66s | 4.4036 | 2.9172 | 1.0 (100%) | 1.0 (100%) | 1,311.25 MB | Second smoke verification run. Produced current checkpoints. |

### Critical Distinction:
- **Smoke Training vs. Real Model Evaluation:**  
  The recorded runs processed only **100 batches (200 video sequences)**.
  - Initial CTC Loss: ~5.59
  - Final CTC Loss: ~4.40
  - Validation CTC Loss: ~2.92
  - Validation CER / WER: **100% (1.0)**
- This confirmed hardware compatibility, AMP stability, gradient flow without NaNs, and MLflow logging. **No converged training or real speech evaluation has occurred.**

---

## 5. Unit Test Suite Results

The test suite was executed in read-only mode using pytest in the project's virtual environment:

- **Command:** `.\.venv\Scripts\python.exe -m pytest tests/ -v`
- **Total Test Cases:** **72**
- **Passed:** **72**
- **Failed:** **0**
- **Errors:** **0**
- **Skipped:** **0**
- **Execution Time:** **68.49 seconds**

### Test Module Breakdown:
1. `tests/test_egcllc.py` (8 tests): GRID alignment parsing, sample indexing, speaker-aware split disjointness, custom collation padding. — **ALL PASSED**
2. `tests/test_environment.py` (7 tests): Python/PyTorch environment, CUDA detection, path resolution, vocabulary encoding/decoding. — **ALL PASSED**
3. `tests/test_inference.py` (27 tests): Greedy decoding, Prefix Beam Search decoding, confidence calculation, error handling, batch decoding, InferenceEngine. — **ALL PASSED**
4. `tests/test_model.py` (9 tests): 3D CNN + BiLSTM dimensions, CTC layout conversions, GPU forward pass, parameter counts, finite CTC loss calculation. — **ALL PASSED**
5. `tests/test_preprocessing.py` (14 tests): Video reader, FaceMesh landmarks, lip ROI cropping, sequence padding/truncation, normalization. — **ALL PASSED**
6. `tests/test_training.py` (7 tests): Optimizer/scheduler creation, seed reproducibility, CER/WER edit distance calculation, checkpoint serialization, gradient accumulation, real backpropagation weight updates. — **ALL PASSED**

---

## 6. Environment & Hardware Profile

- **Operating System:** Windows 11 (AMD64)
- **Python Version:** `3.12.4` (MSC v.1940 64-bit)
- **PyTorch Version:** `2.14.1+cu126`
- **TorchVision Version:** `0.29.1+cu126`
- **CUDA Available:** `True` (CUDA 12.6 runtime)
- **Device Count:** `1`
- **GPU Model:** **NVIDIA GeForce GTX 1650**
- **Total Dedicated VRAM:** **4.0 GB (4,096 MB)**
- **Observed Peak VRAM during Training:** **1,311.25 MB (~1.28 GB)**
- **Headroom:** Sufficient headroom (~2.7 GB free) allows training with batch size 2 and gradient accumulation steps 2 to 4.

---

## 7. Git Repository Status

- **Git Repository Initialized:** **NO** (`Test-Path .git` returns `False`; `git status` reports `fatal: not a git repository`).
- **Current Branch:** None (Not a git repository).
- **Modified / Untracked Files:** The entire workspace directory is currently unversioned locally.
- **Ignored / Configuration Files:** A comprehensive `.gitignore` file (1,462 bytes) is already present and properly configured to ignore `.venv/`, `__pycache__/`, datasets (`E:/LipRead_Data`), checkpoints, and local caches.

---

## 8. Current Implemented Architecture

Reconstructed strictly from active source code:

```
[ Input: Raw Video File (.mp4) or Cached Tensor (.pt) ]
                        │
                        ▼
   ┌────────────────────────────────────────────────────────┐
   │             1. PREPROCESSING PIPELINE                  │
   │  src/preprocessing/                                    │
   │  • VideoReader: OpenCV frame extraction (25 FPS)       │
   │  • FaceLandmarks: MediaPipe FaceLandmarker (478 pts)   │
   │  • LipROIExtractor: Bounding box crop -> 88x88 gray    │
   │  • Pixel Normalization: Scaled to [0.0, 1.0]           │
   │  • SequenceBuilder: Padded / Truncated to T=75         │
   │    Output shape: [T=75, C=1, H=88, W=88]               │
   └────────────────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────────────────┐
   │             2. DATASET & DATALOADER                    │
   │  src/data/                                             │
   │  • EGCLLCDataset: Lazy disk frame loader + cache       │
   │  • VocabularyConfig: 30 classes (blank=0, 'a'-'z', ' ')│
   │  • pad_collate_fn: Batches video [B, 75, 1, 88, 88]    │
   │    and target label sequences [B, max_target_len]      │
   └────────────────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────────────────┐
   │             3. 3D CNN FRONT-END                        │
   │  src/models/cnn3d.py (CNN3D)                           │
   │  • Input: [B, C=1, T=75, H=88, W=88]                   │
   │  • Conv3D-1: 32 filters, 3x5x5 + BN + ReLU + MaxPool   │
   │  • Conv3D-2: 64 filters, 3x3x3 + BN + ReLU + MaxPool   │
   │  • Conv3D-3: 96 filters, 3x3x3 + BN + ReLU + MaxPool   │
   │  • Spatial Pool: AdaptiveAvgPool3d((None, 1, 1))       │
   │    Output shape: [B, T=75, feature_dim=96]             │
   └────────────────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────────────────┐
   │             4. BIDIRECTIONAL LSTM BACK-END             │
   │  src/models/bilstm.py (BiLSTM)                         │
   │  • Input: [B, T=75, 96]                                │
   │  • 2-layer Bi-LSTM (hidden size = 256 per dir)         │
   │  • Inter-layer Dropout: 0.2                            │
   │    Output shape: [B, T=75, 512]                        │
   └────────────────────────────────────────────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────────────────┐
   │             5. LINEAR PROJECTION                       │
   │  src/models/lipread_model.py                           │
   │  • Linear(512, vocab_size=30)                          │
   │    Output Logits: [B, T=75, 30]                        │
   └────────────────────────────────────────────────────────┘
            │                               │
   (Training Branch)               (Inference Branch)
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│       6. CTC LOSS     │       │     7. CTC DECODER    │
│  src/models/ctc.py    │       │  src/inference/       │
│  • Transpose: [T,B,30]│       │  • Greedy best-path   │
│  • torch.nn.CTCLoss   │       │  • Prefix Beam Search │
│  • zero_infinity=True │       │  • Token deduplication│
│  • AdamW + AMP + Clip │       │  • Blank stripping    │
└───────────────────────┘       └───────────────────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │  8. INFERENCE ENGINE  │
                                │  src/inference/       │
                                │  • Prediction output  │
                                │  • Confidence score   │
                                └───────────────────────┘
```

### Additional Implemented Application Layers:
- **CLI Utilities:** Implemented for preprocessing (`src/preprocessing/cli.py`), dataset indexing (`src/data/egcllc_cli.py`), and inference (`src/inference/cli.py`).
- **Web App / Serving / LLM Layers:** **None.** FastAPI, Streamlit, LangChain, and Docker are **not implemented**.

---

## 9. Project Completion Matrix

| Component | Status | Evidence |
|---|---|---|
| **Environment** | **IMPLEMENTED** | Python 3.12.4, PyTorch 2.14.1+cu126, torchvision, CUDA active on GTX 1650 (4.0 GB). |
| **Preprocessing** | **IMPLEMENTED** | `src/preprocessing/` (VideoReader, MediaPipe FaceMesh, LipROI, SequenceBuilder). 14/14 unit tests passing. |
| **Dataset** | **PARTIAL** | Core code (`src/data/`) is fully implemented; physical data extraction is partially completed (20/33 speakers extracted in `lip_images`). |
| **DataLoader** | **IMPLEMENTED** | `src/data/collate.py`, `dataloader.py`, `egcllc_dataset.py` with multi-worker support and dynamic padding collation. |
| **3D CNN** | **IMPLEMENTED** | `src/models/cnn3d.py`. 3-stage spatiotemporal Conv3D with temporal preservation. Fully tested. |
| **Bi-LSTM** | **IMPLEMENTED** | `src/models/bilstm.py`. 2-layer Bidirectional LSTM with dropout. Fully tested. |
| **CTC** | **IMPLEMENTED** | `src/models/ctc.py`. Validated CTCLoss, target padding, and layout conversion. |
| **Training** | **IMPLEMENTED** | `src/training/trainer.py`, `train.py`. Mixed precision, gradient accumulation, gradient clipping. |
| **Evaluation** | **IMPLEMENTED** | `src/training/metrics.py`. Character Error Rate (CER) and Word Error Rate (WER) via Levenshtein edit distance. |
| **Checkpointing** | **IMPLEMENTED** | `src/training/checkpoint.py`. Robust state saving (model, optimizer, scheduler, scaler, metadata). |
| **MLflow** | **IMPLEMENTED** | `src/training/trainer.py`. Local MLflow tracking integrated and actively verified in `E:\LipRead_Data\mlruns`. |
| **Inference** | **IMPLEMENTED** | `src/inference/engine.py`, `decoder.py` (greedy + beam search), and `cli.py`. 27/27 unit tests passing. |
| **FastAPI** | **NOT IMPLEMENTED** | `api/__init__.py` exists only as a docstring placeholder. No endpoints or app logic exist. |
| **Streamlit** | **NOT IMPLEMENTED** | `app/__init__.py` exists only as a docstring placeholder. No UI code exists. |
| **LangChain/LLM** | **NOT IMPLEMENTED** | No LLM integration or prompt logic exists anywhere in `src/`. |
| **Docker** | **NOT IMPLEMENTED** | No `Dockerfile` or `docker-compose.yml` present in repository. |
| **CI/CD** | **NOT IMPLEMENTED** | `.github/workflows/` contains only `.gitkeep`. No automated pipeline configurations exist. |
| **Tests** | **IMPLEMENTED** | 72 comprehensive tests across 6 files in `tests/`. All 72 passing. |
| **Documentation** | **PARTIAL** | Comprehensive `README.md` exists, but overstates completion of unimplemented layers (FastAPI, Streamlit, LangChain, Docker). |

---

## 10. GitHub Readiness Assessment

### A. READY TO PUBLISH
- **Core ML Architecture:** 3D CNN + Bi-LSTM PyTorch implementation (`src/models/`).
- **Computer Vision Preprocessing:** MediaPipe + OpenCV lip ROI pipeline (`src/preprocessing/`).
- **Dataset Infrastructure:** EGCLLC adapter, JSONL indexer, collation, and DataLoaders (`src/data/`).
- **Inference & Decoders:** Greedy and Beam Search CTC decoders and Inference Engine (`src/inference/`).
- **Training Harness:** Full PyTorch Trainer with AMP, gradient accumulation, and MLflow (`src/training/`).
- **Unit Tests:** 72 passing test cases (`tests/`).
- **Configuration & Utilities:** Central `config.yaml`, logging, and reproducibility seed utilities (`configs/`, `src/utils/`).

### B. NEEDS SMALL CLEANUP
- **Initialize Local Git Repository:** Run `git init`, stage tracked code, and create the initial commit.
- **Synchronize README.md with Actual Project Reality:** Update README badges and text to accurately state that the project is currently in the **Core ML / Model Pipeline Verified** phase.
- **Remove or Label Planned Sections:** Clearly mark FastAPI, Streamlit, LangChain/LLM, Docker, and GitHub Actions as **"Planned / Under Construction"** rather than present features.

### C. NOT READY / MUST NOT CLAIM
- **DO NOT CLAIM a Trained Model:** The model weights in `checkpoints/` have only undergone a 100-batch smoke test. They cannot perform accurate lip reading.
- **DO NOT CLAIM Meaningful CER / WER:** Current recorded CER and WER are 1.0 (100%).
- **DO NOT CLAIM Web / API Deployment:** FastAPI and Streamlit interfaces do not exist yet.
- **DO NOT CLAIM LLM Post-Processing:** LangChain integration does not exist yet.
- **DO NOT CLAIM Containerization:** No Docker image or Dockerfile exists.

---

## 11. Critical Limitations to Disclose

1. **Incomplete Dataset Extraction:** The raw EGCLLC dataset extraction was interrupted when the laptop powered off. Only 20 speakers are fully extracted in `lip_images/`, 1 is partially extracted (`s29`), and 12 speakers are missing from disk.
2. **Smoke-Trained Model Only:** The existing checkpoints (`best_model.pt` and `latest_model.pt`) were produced during a 100-step smoke test to verify gradient propagation and AMP stability. The model has not been trained on the full dataset.
3. **No Trained Weights Release:** Any inference run with the current checkpoints will yield random character predictions and 100% CER/WER.
4. **Disjoint Speaker Substitution during Smoke Validation:** Because validation speaker extraction was incomplete when smoke training ran, a test speaker was substituted to satisfy validation loader initialization.
5. **Absence of Serving and Application Layers:** The FastAPI backend, Streamlit dashboard, LangChain correction pipeline, Docker container, and GitHub Actions workflows are planned components that have not yet been developed.

---

## 12. Recommended Next Action

> **Recommendation:**  
> **Resume and complete the EGCLLC dataset extraction for `lip_images.tar`** so that all 33 speakers are fully extracted on disk, and regenerate the dataset split index before initiating full-scale model training.

*(Note: In accordance with audit instructions, this recommendation has NOT been executed).*
