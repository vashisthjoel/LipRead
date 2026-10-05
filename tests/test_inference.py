"""Tests for Stage 6 — Inference Pipeline.

Covers:
  - CTC greedy decoder correctness
  - CTC beam search decoder correctness
  - InferenceEngine instantiation (no checkpoint required)
  - InferenceEngine.transcribe_tensor on random data
  - Batch decoding
  - Edge cases: blank-only sequences, single-frame inputs
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.data.dataset_config import VocabularyConfig
from src.inference.decoder import (
    beam_search_decode,
    ctc_decode,
    ctc_decode_batch,
    greedy_decode,
    greedy_decode_batch,
)
from src.inference.engine import InferenceEngine, InferenceResult
from src.models.lipread_model import LipReadModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vocab() -> VocabularyConfig:
    return VocabularyConfig()


@pytest.fixture()
def small_model(vocab) -> LipReadModel:
    """Tiny model for unit tests — fast to instantiate."""
    return LipReadModel(
        vocab_size=vocab.vocab_size,
        in_channels=1,
        cnn_channels=(8, 16, 16),
        lstm_hidden_size=32,
        lstm_layers=1,
        dropout=0.0,
    )


@pytest.fixture()
def engine(small_model, vocab) -> InferenceEngine:
    return InferenceEngine(
        model=small_model,
        vocab=vocab,
        device="cpu",
        decode_method="greedy",
    )


# ---------------------------------------------------------------------------
# Decoder Tests
# ---------------------------------------------------------------------------


class TestGreedyDecode:
    def test_blank_only_sequence(self, vocab):
        """Blank-only input should decode to empty string."""
        T, V = 10, vocab.vocab_size
        # All probability mass on blank (index 0)
        log_probs = torch.full((T, V), -1e9)
        log_probs[:, vocab.blank_idx] = 0.0
        result = greedy_decode(log_probs, vocab=vocab)
        assert result == "", f"Expected empty string, got '{result}'"

    def test_single_char_repeated(self, vocab):
        """Repeated same character should decode to a single character."""
        T, V = 20, vocab.vocab_size
        # Encode 'a' → char_to_id('a')
        a_idx = vocab.char_to_id("a")
        log_probs = torch.full((T, V), -1e9)
        log_probs[:, a_idx] = 0.0  # Always predict 'a'
        result = greedy_decode(log_probs, vocab=vocab)
        assert result == "a", f"Expected 'a', got '{result}'"

    def test_known_sequence(self, vocab):
        """Sequence [h, h, blank, i] should decode to 'hi'."""
        h_idx = vocab.char_to_id("h")
        i_idx = vocab.char_to_id("i")
        blank = vocab.blank_idx

        T, V = 4, vocab.vocab_size
        log_probs = torch.full((T, V), -1e9)
        log_probs[0, h_idx] = 0.0  # h
        log_probs[1, h_idx] = 0.0  # h (duplicate → collapsed)
        log_probs[2, blank] = 0.0  # blank
        log_probs[3, i_idx] = 0.0  # i
        result = greedy_decode(log_probs, vocab=vocab)
        assert result == "hi", f"Expected 'hi', got '{result}'"

    def test_batch_shape_squeezed(self, vocab):
        """Shape [1, T, V] should be accepted by greedy_decode."""
        T, V = 5, vocab.vocab_size
        log_probs = torch.randn(1, T, V)
        result = greedy_decode(log_probs, vocab=vocab)
        assert isinstance(result, str)

    def test_invalid_batch_size_raises(self, vocab):
        T, V = 5, vocab.vocab_size
        with pytest.raises(ValueError):
            greedy_decode(torch.randn(2, T, V), vocab=vocab)

    def test_invalid_ndim_raises(self, vocab):
        with pytest.raises(ValueError):
            greedy_decode(torch.randn(5), vocab=vocab)


class TestGreedyDecodeBatch:
    def test_batch_output_length(self, vocab):
        B, T, V = 4, 10, vocab.vocab_size
        log_probs = torch.randn(B, T, V)
        results = greedy_decode_batch(log_probs, vocab=vocab)
        assert len(results) == B
        assert all(isinstance(r, str) for r in results)

    def test_single_2d_input(self, vocab):
        T, V = 10, vocab.vocab_size
        log_probs = torch.randn(T, V)
        results = greedy_decode_batch(log_probs, vocab=vocab)
        assert len(results) == 1


class TestBeamSearchDecode:
    def test_blank_only_sequence(self, vocab):
        T, V = 10, vocab.vocab_size
        log_probs = torch.full((T, V), -1e9)
        log_probs[:, vocab.blank_idx] = 0.0
        result = beam_search_decode(log_probs, vocab=vocab, beam_width=5)
        assert result == "", f"Expected empty string, got '{result}'"

    def test_known_sequence_beam_1(self, vocab):
        """Beam-1 should agree with greedy on a clear sequence."""
        h_idx = vocab.char_to_id("h")
        i_idx = vocab.char_to_id("i")
        blank = vocab.blank_idx

        T, V = 4, vocab.vocab_size
        log_probs = torch.full((T, V), -1e9)
        log_probs[0, h_idx] = 0.0
        log_probs[1, h_idx] = 0.0
        log_probs[2, blank] = 0.0
        log_probs[3, i_idx] = 0.0

        greedy_result = greedy_decode(log_probs, vocab=vocab)
        beam_result = beam_search_decode(log_probs, vocab=vocab, beam_width=1)
        assert beam_result == greedy_result, (
            f"Beam-1 '{beam_result}' != greedy '{greedy_result}'"
        )

    def test_beam_squeeze_1_batch_dim(self, vocab):
        T, V = 5, vocab.vocab_size
        log_probs = torch.randn(1, T, V)
        result = beam_search_decode(log_probs, vocab=vocab, beam_width=5)
        assert isinstance(result, str)

    def test_invalid_batch_raises(self, vocab):
        T, V = 5, vocab.vocab_size
        with pytest.raises(ValueError):
            beam_search_decode(torch.randn(2, T, V), vocab=vocab)


class TestCtcDecode:
    def test_greedy_dispatch(self, vocab):
        T, V = 10, vocab.vocab_size
        logits = torch.randn(T, V)
        result = ctc_decode(logits, vocab=vocab, method="greedy")
        assert isinstance(result, str)

    def test_beam_dispatch(self, vocab):
        T, V = 10, vocab.vocab_size
        logits = torch.randn(T, V)
        result = ctc_decode(logits, vocab=vocab, method="beam", beam_width=5)
        assert isinstance(result, str)

    def test_invalid_method_raises(self, vocab):
        T, V = 10, vocab.vocab_size
        with pytest.raises(ValueError):
            ctc_decode(torch.randn(T, V), vocab=vocab, method="invalid_method")

    def test_no_log_softmax(self, vocab):
        """When apply_log_softmax=False, already-log-prob input should work."""
        T, V = 10, vocab.vocab_size
        logits = torch.randn(T, V)
        log_probs = F.log_softmax(logits, dim=-1)
        result = ctc_decode(log_probs, vocab=vocab, method="greedy", apply_log_softmax=False)
        assert isinstance(result, str)


class TestCtcDecodeBatch:
    def test_batch_greedy(self, vocab):
        B, T, V = 3, 10, vocab.vocab_size
        results = ctc_decode_batch(torch.randn(B, T, V), vocab=vocab, method="greedy")
        assert len(results) == B

    def test_batch_beam(self, vocab):
        B, T, V = 2, 8, vocab.vocab_size
        results = ctc_decode_batch(torch.randn(B, T, V), vocab=vocab, method="beam", beam_width=4)
        assert len(results) == B


# ---------------------------------------------------------------------------
# InferenceEngine Tests
# ---------------------------------------------------------------------------


class TestInferenceEngine:
    def test_instantiation(self, engine):
        assert engine is not None
        assert engine.model.training is False  # Must be in eval mode

    def test_transcribe_tensor_greedy(self, engine):
        """Engine should transcribe a random [T, C, H, W] tensor without error."""
        frames = torch.randn(75, 1, 88, 88)
        result = engine.transcribe_tensor(frames)
        assert result.success, f"Inference failed: {result.error}"
        assert isinstance(result.text, str)
        assert result.num_frames == 75
        assert result.decode_method == "greedy"
        assert result.elapsed_ms > 0

    def test_transcribe_tensor_beam(self, engine, vocab):
        """Beam decoding should also produce a result."""
        engine_beam = InferenceEngine(
            model=engine.model,
            vocab=vocab,
            device="cpu",
            decode_method="beam",
            beam_width=5,
        )
        frames = torch.randn(75, 1, 88, 88)
        result = engine_beam.transcribe_tensor(frames)
        assert result.success, f"Inference failed: {result.error}"
        assert isinstance(result.text, str)

    def test_transcribe_tensor_with_batch_dim(self, engine):
        """Engine should handle [1, T, C, H, W] input by squeezing batch dim."""
        frames = torch.randn(1, 75, 1, 88, 88)
        result = engine.transcribe_tensor(frames)
        assert result.success, f"Inference failed: {result.error}"

    def test_transcribe_missing_video(self, engine):
        """Transcribing a non-existent video should return error result, not raise."""
        result = engine.transcribe(video_path="non_existent_video_xyz.mp4")
        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_confidence_is_negative(self, engine):
        """Confidence (mean max log-prob) must be ≤ 0 since it's a log-probability."""
        frames = torch.randn(75, 1, 88, 88)
        result = engine.transcribe_tensor(frames)
        assert result.success
        assert result.confidence <= 0.0, f"Confidence should be ≤ 0, got {result.confidence}"

    def test_override_decode_method(self, engine, vocab):
        """Per-call decode method override should be respected."""
        frames = torch.randn(75, 1, 88, 88)

        engine_beam = InferenceEngine(
            model=engine.model,
            vocab=vocab,
            device="cpu",
            decode_method="greedy",
        )
        result = engine_beam.transcribe_tensor(frames, decode_method="beam", beam_width=4)
        assert result.decode_method == "beam"

    def test_result_str_representation(self, engine):
        """InferenceResult __str__ should not raise."""
        frames = torch.randn(10, 1, 88, 88)
        result = engine.transcribe_tensor(frames)
        s = str(result)
        assert isinstance(s, str)

    def test_single_frame_input(self, engine):
        """Engine should not crash on a very short sequence (T=1)."""
        frames = torch.randn(1, 1, 88, 88)
        result = engine.transcribe_tensor(frames)
        # May succeed or return blank-only output; must not raise
        assert result.error is None or isinstance(result.error, str)
