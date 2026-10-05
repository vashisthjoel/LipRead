"""CTC decoding algorithms for LipRead.

Provides:
  - greedy_decode: Fast best-path decoding (O(T·V))
  - beam_search_decode: Prefix beam search decoding with configurable width
  - ctc_decode: Unified dispatcher that calls either algorithm

All functions accept raw logits [T, V] or [B, T, V] and return decoded
character strings, handling blank collapsing and repeated token merging
correctly per the CTC specification.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.data.dataset_config import VocabularyConfig
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.inference.decoder")


# ---------------------------------------------------------------------------
# Greedy (Best-Path) Decoder
# ---------------------------------------------------------------------------


def greedy_decode(
    log_probs: torch.Tensor,
    vocab: Optional[VocabularyConfig] = None,
    blank_idx: int = 0,
) -> str:
    """Decodes a single CTC log-probability sequence with greedy best-path.

    At every time step, picks the character with the highest log-probability,
    then collapses consecutive duplicates and strips blank tokens.

    Args:
        log_probs: Log-probabilities of shape [T, V]. If 3-D [B, T, V] with
                   B=1, the batch dimension is squeezed automatically.
        vocab: Vocabulary configuration.  Defaults to ``VocabularyConfig()``.
        blank_idx: CTC blank token index (default 0).

    Returns:
        Decoded text string.

    Raises:
        ValueError: If ``log_probs`` has an unsupported shape.
    """
    vocab = vocab or VocabularyConfig()

    if log_probs.ndim == 3:
        if log_probs.shape[0] != 1:
            raise ValueError(
                f"greedy_decode expects a single sequence; got batch size {log_probs.shape[0]}. "
                "Call greedy_decode_batch for multiple sequences."
            )
        log_probs = log_probs.squeeze(0)  # [T, V]

    if log_probs.ndim != 2:
        raise ValueError(
            f"Expected log_probs with shape [T, V] or [1, T, V], got {tuple(log_probs.shape)}."
        )

    # Best token at every timestep  →  [T]
    best_tokens = log_probs.argmax(dim=-1).tolist()

    # CTC collapsing: remove consecutive duplicates then remove blanks
    collapsed: List[int] = []
    prev = None
    for tok in best_tokens:
        if tok != prev:
            collapsed.append(tok)
            prev = tok

    tokens_no_blank = [t for t in collapsed if t != blank_idx]
    return vocab.decode(tokens_no_blank)


def greedy_decode_batch(
    log_probs: torch.Tensor,
    vocab: Optional[VocabularyConfig] = None,
    blank_idx: int = 0,
) -> List[str]:
    """Decodes a batch of CTC sequences with greedy best-path.

    Args:
        log_probs: Log-probabilities of shape [B, T, V] or [T, V] for a single sequence.
        vocab: Vocabulary configuration.
        blank_idx: CTC blank token index.

    Returns:
        List of decoded strings, one per batch element.
    """
    vocab = vocab or VocabularyConfig()

    if log_probs.ndim == 2:
        return [greedy_decode(log_probs, vocab=vocab, blank_idx=blank_idx)]

    if log_probs.ndim != 3:
        raise ValueError(f"Expected [B, T, V] or [T, V], got shape {tuple(log_probs.shape)}")

    return [
        greedy_decode(log_probs[i], vocab=vocab, blank_idx=blank_idx)
        for i in range(log_probs.shape[0])
    ]


# ---------------------------------------------------------------------------
# Beam Search Decoder
# ---------------------------------------------------------------------------


def _logsumexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search_decode(
    log_probs: torch.Tensor,
    vocab: Optional[VocabularyConfig] = None,
    blank_idx: int = 0,
    beam_width: int = 10,
) -> str:
    """Decodes a single CTC log-probability sequence with prefix beam search.

    Implements the standard CTC beam search algorithm (Graves et al. 2006)
    with log-space arithmetic for numerical stability.

    Args:
        log_probs: Log-probabilities of shape [T, V]. If [1, T, V], the batch
                   dimension is squeezed.
        vocab: Vocabulary configuration.
        blank_idx: CTC blank token index.
        beam_width: Number of prefixes to retain at each time step.

    Returns:
        The top-scoring decoded string.

    Raises:
        ValueError: If ``log_probs`` has an unsupported shape.
    """
    vocab = vocab or VocabularyConfig()

    if log_probs.ndim == 3:
        if log_probs.shape[0] != 1:
            raise ValueError(
                f"beam_search_decode expects a single sequence; got batch {log_probs.shape[0]}."
            )
        log_probs = log_probs.squeeze(0)  # [T, V]

    if log_probs.ndim != 2:
        raise ValueError(
            f"Expected log_probs [T, V] or [1, T, V], got {tuple(log_probs.shape)}."
        )

    T, V = log_probs.shape
    # Ensure we are on CPU for the Python-level beam loop
    lp = log_probs.cpu().float()

    # Beam state: dict mapping prefix_tuple -> (log_prob_blank, log_prob_non_blank)
    # p_b  = probability of prefix ending in blank
    # p_nb = probability of prefix ending in non-blank
    NEG_INF = -math.inf
    # Initial empty prefix
    beam: dict[tuple, Tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(T):
        lp_t = lp[t]  # [V]
        new_beam: dict[tuple, Tuple[float, float]] = {}

        # Prune to top beam_width prefixes by total score
        top_prefixes = sorted(
            beam.items(),
            key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            reverse=True,
        )[:beam_width]

        for prefix, (p_b, p_nb) in top_prefixes:
            for c in range(V):
                lp_c = lp_t[c].item()

                if c == blank_idx:
                    # Emit blank → prefix unchanged, ends in blank
                    entry = new_beam.setdefault(prefix, (NEG_INF, NEG_INF))
                    new_p_b = _logsumexp(entry[0], _logsumexp(p_b, p_nb) + lp_c)
                    new_beam[prefix] = (new_p_b, entry[1])
                else:
                    # Emit non-blank character
                    new_prefix = prefix + (c,)
                    entry = new_beam.setdefault(new_prefix, (NEG_INF, NEG_INF))

                    if prefix and prefix[-1] == c:
                        # Same as last char in prefix: only blank path can extend
                        new_p_nb = _logsumexp(entry[1], p_b + lp_c)
                    else:
                        new_p_nb = _logsumexp(entry[1], _logsumexp(p_b, p_nb) + lp_c)

                    new_beam[new_prefix] = (entry[0], new_p_nb)

        beam = new_beam

    # Select best prefix
    best_prefix = max(beam.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    tokens_no_blank = [t for t in best_prefix if t != blank_idx]
    return vocab.decode(tokens_no_blank)


# ---------------------------------------------------------------------------
# Unified Dispatcher
# ---------------------------------------------------------------------------


def ctc_decode(
    logits: torch.Tensor,
    vocab: Optional[VocabularyConfig] = None,
    blank_idx: int = 0,
    method: str = "greedy",
    beam_width: int = 10,
    apply_log_softmax: bool = True,
) -> str:
    """Unified CTC decoding dispatcher for a single sequence.

    Converts raw model logits to a text string via greedy or beam-search
    decoding.

    Args:
        logits: Raw model output logits of shape [T, V] or [1, T, V].
                If 3-D with B=1, the batch dim is squeezed automatically.
        vocab: Vocabulary configuration.
        blank_idx: CTC blank token index.
        method: ``"greedy"`` (default) or ``"beam"``.
        beam_width: Beam width for beam search (ignored for greedy).
        apply_log_softmax: If True (default), applies log_softmax along the
                           last dimension before decoding. Set to False if
                           ``logits`` are already log-probabilities.

    Returns:
        Decoded text string.
    """
    vocab = vocab or VocabularyConfig()

    # Normalise to [T, V]
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"ctc_decode expects B=1, got {logits.shape[0]}")
        logits = logits.squeeze(0)

    log_probs = F.log_softmax(logits, dim=-1) if apply_log_softmax else logits

    if method == "greedy":
        return greedy_decode(log_probs, vocab=vocab, blank_idx=blank_idx)
    elif method == "beam":
        return beam_search_decode(
            log_probs,
            vocab=vocab,
            blank_idx=blank_idx,
            beam_width=beam_width,
        )
    else:
        raise ValueError(f"Unknown decoding method: '{method}'. Choose 'greedy' or 'beam'.")


def ctc_decode_batch(
    logits: torch.Tensor,
    vocab: Optional[VocabularyConfig] = None,
    blank_idx: int = 0,
    method: str = "greedy",
    beam_width: int = 10,
    apply_log_softmax: bool = True,
) -> List[str]:
    """Decodes a batch of sequences.

    Args:
        logits: Logits of shape [B, T, V].
        vocab: Vocabulary configuration.
        blank_idx: CTC blank token index.
        method: ``"greedy"`` or ``"beam"``.
        beam_width: Beam width (beam only).
        apply_log_softmax: Apply log_softmax before decoding.

    Returns:
        List of decoded strings of length B.
    """
    vocab = vocab or VocabularyConfig()

    if logits.ndim == 2:
        logits = logits.unsqueeze(0)

    log_probs = F.log_softmax(logits, dim=-1) if apply_log_softmax else logits

    results = []
    for i in range(log_probs.shape[0]):
        seq = log_probs[i]  # [T, V]
        if method == "greedy":
            results.append(greedy_decode(seq, vocab=vocab, blank_idx=blank_idx))
        else:
            results.append(
                beam_search_decode(seq, vocab=vocab, blank_idx=blank_idx, beam_width=beam_width)
            )
    return results
