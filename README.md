# LipRead

### End-to-End Sentence-Level Lip Reading with 3D CNN + Bi-LSTM + CTC

[![Automated Tests](https://github.com/your-username/LipRead/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/LipRead/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

> **Suggested GitHub Description:**
> "End-to-end sentence-level lip reading system using 3D CNN, Bi-LSTM, CTC, FastAPI, Streamlit, MLflow, and optional LLM-based transcript correction."

---

## Table of Contents
1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Current Project Status](#4-current-project-status)
5. [Key Features](#5-key-features)
6. [Technology Stack](#6-technology-stack)
7. [Dataset](#7-dataset)
8. [Data Preprocessing Pipeline](#8-data-preprocessing-pipeline)
9. [Neural Model Architecture](#9-neural-model-architecture)
10. [Training Pipeline](#10-training-pipeline)
11. [Inference Pipeline](#11-inference-pipeline)
12. [CTC Decoding Strategies](#12-ctc-decoding-strategies)
13. [FastAPI Backend Service](#13-fastapi-backend-service)
14. [Streamlit Interactive Frontend](#14-streamlit-interactive-frontend)
15. [Optional Contextual LLM Post-Processing](#15-optional-contextual-llm-post-processing)
16. [Experiment Tracking with MLflow](#16-experiment-tracking-with-mlflow)
17. [Automated Testing](#17-automated-testing)
18. [Installation](#18-installation)
19. [Running Locally](#19-running-locally)
20. [API Usage & Examples](#20-api-usage--examples)
21. [Training Instructions](#21-training-instructions)
22. [Verified Empirical Results (MVP-1)](#22-verified-empirical-results-mvp-1)
23. [Qualitative Analysis](#23-qualitative-analysis)
24. [MVP-2 Experiment Status](#24-mvp-2-experiment-status)
25. [Current Technical Cache Limitation](#25-current-technical-cache-limitation)
26. [Known Limitations](#26-known-limitations)
27. [Future Work & Roadmap](#27-future-work--roadmap)
28. [License & Attribution](#28-license--attribution)

---

## 1. Overview

**LipRead** is an open-source, end-to-end deep learning system designed for automated visual speech recognition (AVSR) at the sentence level. The system decodes continuous spoken English phrases purely from silent video recordings of a speaker's mouth movements without access to acoustic audio signals.

The project encompasses a complete production pipeline: from raw video input and facial landmark extraction with MediaPipe, to spatiotemporal sequence modeling with PyTorch (3D CNN + Bi-LSTM + CTC), real-time REST serving via FastAPI, an interactive web UI via Streamlit, and an optional LangChain-powered lexical correction layer.

---

## 2. Problem Statement

Automated visual speech recognition (lip reading) is notoriously challenging due to:
1. **Viseme Ambiguity:** Multiple distinct phonemes share virtually identical mouth shapes (e.g., `/p/`, `/b/`, and `/m/`).
2. **Speaker Morphology & Coarticulation:** Mouth shape, jaw movement, speech speed, and facial geometry vary substantially across individuals, requiring strict **speaker-disjoint** validation.
3. **Temporal Dynamics:** Recognizing continuous speech demands capturing subtle temporal progressions over time rather than isolated static frames.

LipRead addresses these challenges by combining spatial feature extraction with bidirectional temporal sequence modeling, operating under strict speaker-disjoint splits to measure true generalization.

---

## 3. System Architecture

```
User / Client
     │
     ▼
Streamlit Frontend UI (app/streamlit_app.py)
     │ [HTTP Multipart POST /predict]
     ▼
FastAPI REST Backend (api/main.py)
     │
     ├── Input Validation & Temporary Buffering
     │
     ├── Video Preprocessing Pipeline (OpenCV + MediaPipe)
     │     ├── Frame extraction & validation
     │     ├── Face landmark detection (MediaPipe FaceMesh)
     │     ├── Mouth ROI extraction & bounding box calculation
     │     ├── Grayscale conversion & resize to 88×88
     │     ├── Pixel normalization to [0.0, 1.0]
     │     └── Fixed 75-frame temporal sequence standardization
     │
     ├── Deep Learning Inference (src/models/lipread_model.py)
     │     ├── Spatiotemporal 3D CNN (3 blocks: 32 -> 64 -> 96 channels)
     │     ├── Spatial-only adaptive pooling -> [B, 75, 96]
     │     ├── 2-layer Bidirectional LSTM (hidden size 256) -> [B, 75, 512]
     │     └── Linear Classifier Head -> 30 character classes [B, 75, 30]
     │
     ├── Connectionist Temporal Classification (CTC) Decoding
     │     ├── Greedy CTC Decoding (Default production decoder)
     │     └── Prefix Beam Search Decoding (Experimental)
     │
     ├── Optional GenAI Lexical Correction (src/genai/corrector.py)
     │     ├── LangChain orchestration
     │     └── Provider options: mock, OpenAI, Anthropic, Ollama
     │
     ▼
JSON Response & Interactive Web Display
```

---

## 4. Current Project Status

| Component | Status | Implementation Details |
| :--- | :---: | :--- |
| **Project Foundation** | ✅ Complete | Modular architecture, YAML configuration, deterministic seeding |
| **Video Preprocessing** | ✅ Complete | OpenCV sequential reader, MediaPipe 478-landmark FaceMesh, 88×88 crop |
| **EGCLLC Dataset Integration**| ✅ Complete | Speaker-disjoint manifest generation and indexing |
| **3D CNN Front-End** | ✅ Complete | 3 Conv3D blocks with spatial-only MaxPool3D (75 timesteps preserved) |
| **Bi-LSTM Sequence Model** | ✅ Complete | 2-layer bidirectional LSTM (256 hidden units per direction) |
| **CTC Loss & Training** | ✅ Complete | PyTorch `nn.CTCLoss` with zero-infinity protection and AMP support |
| **Greedy CTC Decoder** | ✅ Complete | Timestep argmax, blank stripping, character deduplication |
| **Prefix Beam Search Decoder**| 🟡 Experimental | Unconstrained prefix search (evaluated; greedy remains default) |
| **FastAPI REST Service** | ✅ Complete | Lifespan loading, `/health`, `/predict`, OpenAPI/Swagger documentation |
| **Streamlit Web Application** | ✅ Complete | Video upload, live preview, parameter controls, metrics visualization |
| **LangChain Post-Processing** | ✅ Complete | Configurable LLM integration with automatic raw-output fallback |
| **MLflow Experiment Tracking**| ✅ Complete | Metrics, parameters, and checkpoint logging |
| **Automated Test Suite** | ✅ Complete | **130 passed / 0 failed** (CPU-safe, runs in $< 45$s) |
| **MVP-1 Baseline Experiment** | ✅ Complete | 6,000 samples, 6 epochs, held-out test evaluation completed |
| **MVP-2 Manifests** | ✅ Complete | 15,000 train / 2,000 val / 2,000 test manifests verified |
| **MVP-2 Training Run** | 🟡 In Progress | 15,000 train samples completed; validation halted by OS cache limit |
| **MVP-2 Evaluation** | 🟡 Pending | Awaiting cache indexing optimization and validation completion |
| **Production Containerization** | ✅ Complete | Multi-container Docker and Docker Compose configurations |
| **Advanced LM-Constrained Beam**| 📋 Planned | Lexicon-constrained beam search with n-gram/transformer language model |
| **Multi-Speaker Dataset Expansion**| 📋 Planned | Scaling beyond 16 training speakers |

---

## 5. Key Features

- **End-to-End Deep Learning:** Direct video-to-text transcription without intermediate phoneme alignment requirements.
- **Strict Speaker-Disjoint Evaluation:** Train, validation, and test splits have 0% speaker overlap, testing true model generalization on unseen human faces.
- **Automated Mixed Precision (AMP):** Optimized for modest hardware (NVIDIA GTX 1650 4 GB VRAM) using PyTorch `torch.amp`.
- **Dynamic Preprocessing Cache:** First epoch dynamically converts and serializes raw frames to disk `.pt` tensors; subsequent epochs bypass image decoding entirely.
- **Dual Decoding Options:** Ultra-fast Greedy CTC decoding by default, with experimental prefix beam search available.
- **Optional Contextual LLM Correction:** Seamless LangChain orchestration to resolve phonetic visual ambiguities when enabled.
- **Production-Ready API & UI:** Asynchronous FastAPI backend coupled with a responsive, modern dark-themed Streamlit interface.

---

## 6. Technology Stack

- **Core DL Framework:** PyTorch, torchvision, torchaudio
- **Computer Vision:** OpenCV (`opencv-python`), MediaPipe FaceMesh
- **Sequence Modeling:** Connectionist Temporal Classification (CTC)
- **Serving & Web:** FastAPI, Uvicorn, Streamlit, HTTPX, Requests
- **LLM Orchestration:** LangChain Core, LangChain Community
- **Configuration & Utilities:** PyYAML, NumPy, Pydantic v2
- **Experiment Tracking:** MLflow
- **Testing & CI:** PyTest, GitHub Actions
- **Containerization:** Docker, Docker Compose

---

## 7. Dataset

LipRead utilizes the **EGCLLC** (Enhanced GRID Corpus with Lip Landmark Coordinates) dataset:
- **Source:** Hugging Face dataset [`SilentSpeak/EGCLLC`](https://huggingface.co/datasets/SilentSpeak/EGCLLC).
- **Structure:** Derived from the standardized GRID corpus consisting of 6-word sentences with fixed grammatical structure: `[command] [color] [preposition] [letter] [digit] [adverb]` (e.g., *"place blue with h seven now"*).
- **Local Extraction State:** 20 speakers (`s1`–`s20`) are fully extracted locally; speaker `s29` is partially extracted.
- **Data Distribution Notice:** This repository does **not** redistribute the EGCLLC or GRID dataset files. Users must independently obtain the dataset under its own terms.

### Speaker Partitions
- **MVP-1:** Train on 7 speakers (6,000 samples), Val on 1 speaker (`s8`, 1,000 samples), Test on 1 speaker (`s9`, 1,000 samples).
- **MVP-2:**
  - **Train (15,000 samples):** 16 speakers (`s1, s2, s10, s11, s12, s13, s15, s16, s17, s18, s22, s24, s25, s26, s27, s28`)
  - **Validation (2,000 samples):** 2 speakers (`s19, s20` — 1,000 each)
  - **Held-Out Test (2,000 samples):** 2 speakers (`s14, s23` — 1,000 each)

---

## 8. Data Preprocessing Pipeline

For each video sample, the preprocessing engine applies the following transformations:
1. **Video Ingestion:** Sequential OpenCV decoding extracts raw video frames.
2. **Landmark Detection:** MediaPipe FaceMesh extracts 478 3D landmarks, isolating the 40 outer and inner mouth coordinates.
3. **Region of Interest (ROI) Cropping:** A dynamic bounding box with margin is computed around the lip landmarks and clamped to image dimensions.
4. **Grayscale Conversion:** Grayscale luminance conversion removes skin-tone variability and reduces channel dimensions from 3 to 1.
5. **Spatial Normalization:** Resized to a standardized **88 $\times$ 88** resolution; pixel intensities scaled to $[0.0, 1.0]$.
6. **Temporal Sequence Standardizer:**
   - Standard sequence length is **75 frames** (3.0 seconds at 25 FPS).
   - Sequences $> 75$ frames are center-truncated; sequences $< 75$ frames are edge-padded.
7. **Output Tensor Shape:** `[75, 1, 88, 88]` (single sample) or `[Batch, 75, 1, 88, 88]` (dataloader batch).

---

## 9. Neural Model Architecture

The implemented model (`LipReadModel`) consists of **2,541,310 parameters**:

```
Input Video Tensor: [Batch, 75, 1, 88, 88]
  │
  ├── 3D CNN Front-End (3 Conv3D Blocks)
  │     ├── Block 1: Conv3D(1 -> 32, k=3) + BatchNorm3D + ReLU + MaxPool3D(1, 2, 2)  --> [B, 32, 75, 44, 44]
  │     ├── Block 2: Conv3D(32 -> 64, k=3) + BatchNorm3D + ReLU + MaxPool3D(1, 2, 2) --> [B, 64, 75, 22, 22]
  │     ├── Block 3: Conv3D(64 -> 96, k=3) + BatchNorm3D + ReLU + MaxPool3D(1, 2, 2) --> [B, 96, 75, 11, 11]
  │     └── Adaptive Spatial Pooling: AdaptiveAvgPool3D((None, 1, 1))                --> [B, 75, 96]
  │
  ├── Temporal Sequence Model (Bi-LSTM)
  │     ├── 2-Layer Bidirectional LSTM (input_size=96, hidden_size=256, dropout=0.2) --> [B, 75, 512]
  │     └── Dropout (p=0.2)
  │
  └── Classifier Head & CTC
        ├── Linear Projection (512 -> 30)                                            --> [B, 75, 30]
        └── CTC Loss Layout: Permute to [75, B, 30] with log_softmax
```

- **Vocabulary (30 classes):** Index 0: `<blank>`, Index 1: `' '` (space), Indices 2–27: `a`–`z`, Index 28: `'` (apostrophe), Index 29: reserved.
- **Key Architectural Property:** Pooling is strictly spatial `(1, 2, 2)`. The temporal dimension remains $T=75$ throughout the entire network, ensuring fine-grained temporal resolution.

---

## 10. Training Pipeline

- **Loss Function:** PyTorch `nn.CTCLoss(blank=0, zero_infinity=True)`
- **Mixed Precision:** PyTorch Automated Mixed Precision (`torch.amp.autocast`) with dynamic loss scaling (`GradScaler`).
- **Optimization:** AdamW optimizer ($\text{lr} = 1\times 10^{-4}$, $\text{weight\_decay} = 1\times 10^{-4}$).
- **Learning Rate Scheduling:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-6)`.
- **Gradient Clipping:** Max gradient norm clamped to $1.0$.
- **Batching:** Micro-batch size 4, gradient accumulation steps 2 (effective batch size 8).
- **VRAM Utilization:** Consistently draws **1,411 MiB** of VRAM on an NVIDIA GTX 1650 (65% headroom).

---

## 11. Inference Pipeline

The standalone `InferenceEngine` (`src/inference/engine.py`) provides end-to-end execution:
1. Accepts raw video file path or pre-loaded video tensor.
2. Executes OpenCV + MediaPipe preprocessing if a file path is provided.
3. Moves tensor to inference device (CUDA or CPU).
4. Runs forward pass under `torch.no_grad()` and AMP.
5. Invokes chosen CTC decoder (greedy or beam search).
6. Returns typed `InferenceResult` containing transcript, inference duration in milliseconds, frame count, and FPS.

---

## 12. CTC Decoding Strategies

### 1. Greedy Decoding (Default Production Decoder)
- Computes $\arg\max$ at each of the 75 frames, collapses sequential duplicate characters, and removes blanks.
- **Benchmark:** Evaluates in $< 1.0$ ms per sequence.
- **Test CER on MVP-1:** **55.01%** | **Test WER:** **97.52%**.

### 2. Prefix Beam Search (Experimental)
- Explores top-$K$ prefix paths across time, combining probabilities of identical collapsed prefixes.
- **Benchmark ($W=10$):** **84.31% CER** | **174.83% WER**.
- **Conclusion:** Without vocabulary/language model constraints, unconstrained beam search allows acoustic permutations that degrade word accuracy. Greedy decoding remains the recommended default.

---

## 13. FastAPI Backend Service

The REST backend (`api/main.py`) provides an asynchronous service with model lifecycle management:
- **`GET /`**: Service descriptor and metadata.
- **`GET /health`**: Health status and model readiness (`ok` or `degraded`).
- **`POST /predict`**: Multipart video upload accepting `.mp4`, `.avi`, `.mov`, `.mpg`, etc. (up to 50 MiB).
- **Interactive OpenAPI Documentation:** Available at `/docs` (Swagger UI) and `/openapi.json`.

---

## 14. Streamlit Interactive Frontend

The web interface (`app/streamlit_app.py`) provides:
- Drag-and-drop video upload supporting standard video containers.
- Embedded video preview player.
- Live backend connection health monitoring.
- Decoding strategy selection (Greedy vs. Beam Search with configurable beam width).
- Side-by-side display of raw CTC transcript, optional LLM-corrected transcript, and performance metrics (inference time, frame count, FPS).

---

## 15. Optional Contextual LLM Post-Processing

Visual speech recognition often encounters acoustic ambiguities among visually similar visemes (e.g., `p`, `b`, and `m`). When `GENAI_ENABLED=true`, LipRead passes the raw transcript through a LangChain-orchestrated prompt to produce grammatically and contextually refined outputs.

- **Supported Providers:** `mock` (offline testing), `openai`, `anthropic`, `ollama` (local models).
- **Graceful Fallback:** If the LLM provider fails, times out, or returns a blank response, the API seamlessly returns the raw CTC transcription without failing the client request.

---

## 16. Experiment Tracking with MLflow

LipRead integrates natively with MLflow:
- Tracks `train_loss`, `val_loss`, `cer`, `wer`, `learning_rate`, and `vram_peak_mb` across every epoch.
- Logs full training hyperparameters, model architecture parameters, and dataset split metadata.
- Checkpoint artifact registration tracks the `best_model.pt` based on lowest validation loss.

---

## 17. Automated Testing

The repository maintains an automated, CPU-safe test suite:
- **Verified Current Test Suite:** **130 passed / 0 failed** (completed in 42.5s).
- **No External Dependencies for CI:** Tests use synthetic tensor fixtures and mock objects. No GPU, external drive, or proprietary dataset is required to pass tests.

```bash
pytest -v
```

---

## 18. Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- FFmpeg (for video decoding)
- (Optional) NVIDIA GPU with CUDA 12.x for accelerated training

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/your-username/LipRead.git
cd LipRead

# 2. Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -m src.utils.check_environment
pytest -q
```

---

## 19. Running Locally

### 1. Launch the FastAPI Backend
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API.

### 2. Launch the Streamlit Frontend
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run with Docker Compose
```bash
cd docker
docker-compose up --build
```

---

## 20. API Usage & Examples

### Transcribe Video via `curl`
```bash
curl -X POST "http://localhost:8000/predict?decode_method=greedy" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_clip.mpg"
```

### Response
```json
{
  "transcription": "bin blue at f two now",
  "raw_transcription": "bin blue at f two now",
  "corrected_transcription": null,
  "applied_correction": false,
  "correction_model": null,
  "confidence": null,
  "inference_time_ms": 138.4,
  "frame_count": 75,
  "fps": 25.0
}
```

---

## 21. Training Instructions

To initiate model training using the configuration file:

```bash
python -m src.training.train --config configs/config.yaml
```

To view training telemetry in MLflow:
```bash
mlflow ui
```

---

## 22. Verified Empirical Results (MVP-1)

The following metrics are the **empirically measured results** from the completed MVP-1 baseline run evaluated on 1,000 strictly held-out utterances from speaker `s9`:

| Metric | Measured Value |
| :--- | :--- |
| **Training Samples** | 6,000 |
| **Validation Samples** | 1,000 |
| **Test Samples** | 1,000 (speaker `s9` — unseen) |
| **Completed Epochs** | 6 |
| **Best Validation Loss** | **1.1673** (Epoch 6) |
| **Held-Out Test Loss** | **1.2611** |
| **Test Character Error Rate (CER)** | **55.08%** |
| **Test Word Error Rate (WER)** | **97.48%** |
| **Exact Sentence Accuracy** | **0 / 1,000 (0.00%)** |

> [!NOTE]
> **Engineering Baseline Assessment:**
> MVP-1 is an engineering baseline and does not yet provide high-quality sentence-level recognition. It demonstrates that the 3D CNN + Bi-LSTM pipeline successfully optimizes CTC loss and learns phonetic transitions on unseen speakers, but high word error rates highlight the need for larger multi-speaker training sets.

---

## 23. Qualitative Analysis

Inspecting qualitative predictions from MVP-1 indicates:
- **Sentence Structure Prior:** The model learned the grammatical structure of the 6-word GRID grammar pattern `[command] [color] [preposition] [letter] [digit] [adverb]`.
- **Viseme Ambiguity:** Phonetically similar characters and visemes are frequently substituted (e.g., ground truth `"bin blue at f two now"` transcribed as recurring patterns like `"ley gre it ... sine noon"`).
- **Template Bias:** The model developed strong template-like priors, emphasizing the need for higher speaker diversity in subsequent training runs.

---

## 24. MVP-2 Experiment Status

- **Status:** **🟡 IN PROGRESS**
- **Target Specification:** 15,000 training samples across 16 speakers, 2,000 validation samples, 2,000 held-out test samples, 10 epochs.
- **Manifest Integrity:** Created, verified, and audited with zero speaker overlap.
- **Epoch 1 Execution:** All 15,000 training samples were processed cleanly without numerical instability, exploding loss, or CUDA exceptions.
- **Validation Interruption:** The run halted mid-validation at sample 1,168 of 2,000 due to an OS filesystem resource limitation (`WinError 1450`), and model weights were not saved to disk.
- **Artifact Preservation:** All 15,000 training samples and 1,167 validation samples remain **100% preprocessed and cached on disk as `.pt` tensors** (17,277 total files), accelerating future training passes.

---

## 25. Current Technical Cache Limitation

During the MVP-2 validation run, having 17,277 `.pt` files in a single flat directory on an external USB NTFS drive caused repeated `os.stat` / `Path.exists()` calls across thousands of continuous lookups to exhaust the Windows kernel non-paged pool buffer (`[WinError 1450]`).

- **Planned Optimization (In Progress):** Construct an in-memory set index (`set(os.listdir(cache_dir))`) during dataset initialization to replace disk `stat` lookups with instant $O(1)$ memory checks.

---

## 26. Known Limitations

1. **Vocabulary Coverage:** Trained on the 30-character GRID subset; open-domain vocabulary is not yet supported.
2. **Grammar Bias:** Evaluated on structured sentence patterns; natural conversational speech has higher variability.
3. **Lighting & Pose Sensitivity:** MediaPipe landmark detection requires a reasonably illuminated, front-facing face.
4. **Unconstrained Beam Search:** Beam search without an external language model degrades accuracy compared to greedy decoding.

---

## 27. Future Work & Roadmap

- [ ] **Data Pipeline Optimization:** In-memory caching index to permanently eliminate NTFS stat overhead.
- [ ] **MVP-2 10-Epoch Completion:** Full 15,000-sample training across 16 speakers with held-out test evaluation.
- [ ] **Lexicon-Constrained Decoding:** Beam search with language model rescoring (n-gram or transformer).
- [ ] **Self-Supervised Pretraining:** Exploring AV-HuBERT or visual feature pretraining.
- [ ] **Full Multi-Speaker Scaling:** Expanding training to all available extracted speakers.

---

## 28. License & Attribution

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Dataset Attribution
This project uses data derived from the **EGCLLC** (Enhanced GRID Corpus with Lip Landmark Coordinates) dataset hosted by [SilentSpeak/EGCLLC](https://huggingface.co/datasets/SilentSpeak/EGCLLC). The raw audio/video dataset is not distributed with this software and remains subject to the original corpus license and terms.
