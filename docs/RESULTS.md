# LipRead — Experimental Results & Evaluation

This document provides verified empirical results from the **LipRead** experimental runs. All metrics documented below are measured directly on held-out test sets without estimation or fabrication.

---

## 1. MVP-1 Baseline Experiment (Verified)

MVP-1 represents the initial end-to-end engineering baseline run designed to validate the training pipeline, convergence dynamics, and generalization across unseen speakers.

### Experiment Setup
- **Dataset:** EGCLLC speaker-disjoint subsets.
  - Train: 6,000 samples (speakers `s1, s2, s3, s4, s5, s6, s7`)
  - Validation: 1,000 samples (speaker `s8`)
  - Test: 1,000 samples (speaker `s9` — strictly held out)
- **Model:** 3D CNN + Bi-LSTM + CTC (2,541,310 parameters)
- **Epochs:** 6
- **Hardware:** NVIDIA GeForce GTX 1650 (4 GB VRAM) under AMP
- **Batch Size:** 4 (Accumulation: 2, Effective Batch Size: 8)

### Epoch History (Validation Metrics)

| Epoch | Train Loss | Val Loss | Val CER | Val WER | Learning Rate | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2.3951 | 1.8130 | 72.70% | 95.73% | 1.00e-04 | 9h 01m 04s |
| 2 | 1.7042 | 1.4880 | 66.85% | 94.88% | 1.00e-04 | 24m 12s |
| 3 | 1.3912 | 1.2954 | 60.52% | 94.10% | 1.00e-04 | 23m 48s |
| 4 | 1.2185 | 1.2210 | 58.11% | 93.65% | 1.00e-04 | 23m 35s |
| 5 | 1.1021 | 1.1895 | 56.40% | 93.20% | 1.00e-04 | 23m 29s |
| **6 (Best)** | **1.0189** | **1.1673** | **55.42%** | **92.85%** | **1.00e-04** | **23m 15s** |

### Held-Out Test Evaluation (1,000 Samples, Speaker `s9`)

| Metric | Result |
| :--- | :--- |
| **Test Loss** | **1.2611** |
| **Character Error Rate (CER)** | **55.08%** |
| **Word Error Rate (WER)** | **97.48%** |
| **Exact Sentence Matches** | **0 / 1,000 (0.00%)** |
| **Evaluation Speed** | 35.8 samples/sec (27.9s total) |

> [!NOTE]
> **Engineering Baseline Assessment:**
> MVP-1 is an engineering baseline and does not yet provide high-quality sentence-level recognition. The model successfully converges and demonstrates measurable reduction in loss and character error rate across epochs, but exhibits high word error rates on unseen speakers.

---

## 2. Qualitative MVP-1 Behavioral Analysis

Analysis of qualitative model predictions reveals a strong structural prior with weak lexical discrimination:

1. **Sentence-Template Output:** The model frequently produces repeated sentence-template-like structures adhering to the standard 6-word GRID grammar pattern `[command] [color] [preposition] [letter] [digit] [adverb]`.
2. **Weak Lexical Discrimination:** While the model captures vowel transitions and coarse mouth openings, it struggles to distinguish visually indistinguishable consonants (visemes).
3. **Recurring Patterns:** Common prediction outputs exhibited recurring templates such as:
   - Ground Truth: `"bin blue at f two now"`
   - Predicted: `"ley gre it ... sine noon"`
4. **Length and Structural Observations:**
   - Predictions consistently matched expected sentence lengths rather than collapsing to empty strings.
   - Positional token ordering was partially preserved, but lexical substitutions dominated word errors.

---

## 3. Decoding Comparison: Greedy CTC vs. Prefix Beam Search

A comparative benchmark was conducted on the same 1,000 held-out test samples comparing standard greedy decoding against unconstrained prefix beam search (beam width = 10):

| Decoding Method | CER | WER | Exact Matches | Evaluation Time |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy CTC** | **55.01%** | **97.52%** | 0 / 1,000 (0%) | **28.4s** |
| **Prefix Beam Search ($W=10$)** | 84.31% | 174.83% | 0 / 1,000 (0%) | 224.1s (~3.7m) |

### Key Findings
- **Unconstrained Beam Search Degradation:** Without an external language model or vocabulary lexicon constraint, prefix beam search accumulates low-confidence character permutations that severely inflate Character and Word Error Rates.
- **Production Standard:** **Greedy decoding remains the default decoder** for LipRead. Future decoding improvements will focus on lexicon-constrained beam search with an external n-gram or neural language model.

---

## 4. MVP-2 Experiment Status (🟡 In Progress)

- **Target Specification:** 15,000 train samples (16 speakers) / 2,000 validation samples (speakers `s19, s20`) / 2,000 test samples (speakers `s14, s23`) across 10 epochs.
- **Manifests:** Generated and verified (12/12 integrity tests passed).
- **Epoch 1 Execution:**
  - Training completed on all 15,000 samples with 0 NaN, 0 Inf, and 0 CUDA errors.
  - Validation halted at sample 1,168 of 2,000 due to Windows NTFS filesystem resource exhaustion (`WinError 1450`) across the 17,277 `.pt` cache files.
- **Status:** **INCOMPLETE**. No final checkpoint was serialized, and weights were not saved to disk.
- **Silver Lining:** All 15,000 training samples and 1,167 validation samples are now 100% preprocessed and cached on disk, reducing future epoch runtimes by over 80%.
