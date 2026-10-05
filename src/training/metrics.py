"""Evaluation metrics for LipRead.

Provides standard Automatic Speech Recognition (ASR) metrics:
- Levenshtein Edit Distance (Wagner-Fischer algorithm)
- Character Error Rate (CER)
- Word Error Rate (WER)

No external heavy APIs are required. Handles edge cases such as empty
predictions and silent audio transcripts gracefully.
"""

from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")


def compute_edit_distance(reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    """Computes the minimum edit distance (Levenshtein distance) between two sequences.

    Supports insertions, deletions, and substitutions with equal unit cost.
    Optimized for memory with O(min(len(reference), len(hypothesis))) space.

    Args:
        reference: Ground truth sequence of elements (tokens, characters, or words).
        hypothesis: Predicted sequence of elements.

    Returns:
        Integer minimum number of edit operations required.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return m
    if m == 0:
        return n

    # Ensure reference is the shorter sequence to minimize buffer memory
    if n > m:
        reference, hypothesis = hypothesis, reference  # type: ignore
        n, m = m, n

    current_row = list(range(n + 1))
    for i in range(1, m + 1):
        prev_row = current_row
        current_row = [i] + [0] * n
        hyp_elem = hypothesis[i - 1]
        for j in range(1, n + 1):
            ref_elem = reference[j - 1]
            cost = 0 if ref_elem == hyp_elem else 1
            current_row[j] = min(
                prev_row[j] + 1,        # deletion from hypothesis
                current_row[j - 1] + 1,    # insertion into hypothesis
                prev_row[j - 1] + cost    # substitution
            )

    return current_row[n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes the Character Error Rate (CER) between a reference and a hypothesis.

    CER = Levenshtein(ref, hyp) / max(len(ref), 1)

    Args:
        reference: Ground truth text string.
        hypothesis: Predicted text string.

    Returns:
        Character error rate as a float (0.0 means perfect match).
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    if not ref_chars and not hyp_chars:
        return 0.0
    if not ref_chars:
        return 1.0

    distance = compute_edit_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Computes the Word Error Rate (WER) between a reference and a hypothesis.

    WER = Levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)

    Args:
        reference: Ground truth text string.
        hypothesis: Predicted text string.

    Returns:
        Word error rate as a float (0.0 means perfect match).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0

    distance = compute_edit_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def compute_batch_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Computes corpus-level Character Error Rate aggregated across multiple samples.

    Standard ASR definition:
        Total Char Edit Distance / Total Reference Characters

    Args:
        references: List of ground-truth strings.
        hypotheses: List of predicted strings.

    Returns:
        Aggregated CER float.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"References count ({len(references)}) must equal hypotheses count ({len(hypotheses)})."
        )
    if not references:
        return 0.0

    total_distance = 0
    total_ref_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        total_distance += compute_edit_distance(ref_chars, hyp_chars)
        total_ref_len += len(ref_chars)

    if total_ref_len == 0:
        return 0.0 if total_distance == 0 else 1.0

    return total_distance / total_ref_len


def compute_batch_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Computes corpus-level Word Error Rate aggregated across multiple samples.

    Standard ASR definition:
        Total Word Edit Distance / Total Reference Words

    Args:
        references: List of ground-truth strings.
        hypotheses: List of predicted strings.

    Returns:
        Aggregated WER float.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"References count ({len(references)}) must equal hypotheses count ({len(hypotheses)})."
        )
    if not references:
        return 0.0

    total_distance = 0
    total_ref_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_distance += compute_edit_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)

    if total_ref_words == 0:
        return 0.0 if total_distance == 0 else 1.0

    return total_distance / total_ref_words
