"""Unit and integration tests for LipRead model architecture (Stage 4).

Validates:
  - Model construction and layer initializations
  - Vocabulary size matching project VocabularyConfig
  - Total and trainable parameter counting utilities
  - Forward pass on CPU
  - Forward pass on CUDA (if available)
  - Output logits shape [B, 75, V]
  - CTC layout transposition [75, B, V]
  - CTC loss calculation with synthetic targets and finiteness check
  - Input dimension validation (rejects invalid 4D tensors or wrong channel dimensions)
  - Logits numeric stability (no NaN or Inf values)
  - Batch size 1 and batch size 2 execution
  - Greedy CTC decoding collapse and blank removal
"""

from __future__ import annotations

import unittest
import torch

from src.data.dataset_config import VocabularyConfig
from src.models.bilstm import BiLSTM
from src.models.cnn3d import CNN3D
from src.models.ctc import CTCLossWrapper, decode_to_text, greedy_decode
from src.models.lipread_model import LipReadModel
from src.models.model_utils import (
    count_parameters,
    count_trainable_parameters,
    get_model_summary,
    to_ctc_layout,
    to_device,
    to_log_probs,
)


class TestModelArchitecture(unittest.TestCase):
    """Tests 3D CNN, Bi-LSTM, and LipReadModel construction and execution."""

    def setUp(self) -> None:
        self.vocab = VocabularyConfig()
        self.expected_vocab_size = self.vocab.vocab_size  # 30
        self.model = LipReadModel(vocab_size=self.expected_vocab_size)
        self.model.eval()

    def test_model_construction_and_vocab_size(self) -> None:
        """Verifies model initializes with correct vocabulary size and submodules."""
        self.assertIsInstance(self.model.frontend, CNN3D)
        self.assertIsInstance(self.model.backend, BiLSTM)
        self.assertEqual(self.model.vocab_size, self.expected_vocab_size)
        self.assertEqual(self.model.classifier.out_features, self.expected_vocab_size)
        self.assertEqual(self.model.classifier.in_features, 512)

    def test_parameter_count_utilities(self) -> None:
        """Verifies parameter counting utilities return positive, matching counts."""
        total = count_parameters(self.model)
        trainable = count_trainable_parameters(self.model)

        self.assertGreater(total, 1_000_000)
        self.assertEqual(total, trainable)  # All parameters trainable by default

        summary = get_model_summary(self.model)
        self.assertEqual(summary["total_parameters"], total)
        self.assertIn("frontend", summary["components"])
        self.assertIn("backend", summary["components"])
        self.assertIn("classifier", summary["components"])

    def test_cpu_forward_pass_batch_1(self) -> None:
        """Verifies forward pass runs on CPU with batch_size=1 and outputs [1, 75, V]."""
        x = torch.randn((1, 75, 1, 88, 88), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)

        self.assertEqual(tuple(logits.shape), (1, 75, self.expected_vocab_size))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_cpu_forward_pass_batch_2(self) -> None:
        """Verifies forward pass runs on CPU with batch_size=2 and outputs [2, 75, V]."""
        x = torch.randn((2, 75, 1, 88, 88), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)

        self.assertEqual(tuple(logits.shape), (2, 75, self.expected_vocab_size))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_cuda_forward_pass_if_available(self) -> None:
        """Verifies forward pass on CUDA device if a GPU is detected."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this machine.")

        model_cuda, device = to_device(self.model, "cuda")
        x = torch.randn((1, 75, 1, 88, 88), dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model_cuda(x)

        self.assertEqual(tuple(logits.shape), (1, 75, self.expected_vocab_size))
        self.assertEqual(logits.device.type, "cuda")
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        del x, logits
        torch.cuda.empty_cache()

    def test_ctc_layout_conversion(self) -> None:
        """Verifies to_ctc_layout transposes [B, T, V] to [T, B, V]."""
        logits = torch.randn((2, 75, self.expected_vocab_size))
        ctc_logits = LipReadModel.to_ctc_layout(logits)

        self.assertEqual(tuple(ctc_logits.shape), (75, 2, self.expected_vocab_size))
        # Ensure values correspond to transposition without copying
        self.assertTrue(torch.equal(ctc_logits[0, 1], logits[1, 0]))

    def test_invalid_input_dimensions_raise_error(self) -> None:
        """Verifies model raises descriptive ValueError on invalid dimensions."""
        # 4D tensor instead of 5D
        x_4d = torch.randn((1, 75, 88, 88))
        with self.assertRaises(ValueError) as ctx:
            self.model(x_4d)
        self.assertIn("Expected 5D input tensor", str(ctx.exception))

        # 3 channels instead of 1
        x_wrong_c = torch.randn((1, 75, 3, 88, 88))
        with self.assertRaises(ValueError) as ctx:
            self.model(x_wrong_c)
        self.assertIn("Expected channel dimension", str(ctx.exception))


class TestCTCLossAndDecoding(unittest.TestCase):
    """Tests CTCLossWrapper, greedy decoding, and character mapping."""

    def setUp(self) -> None:
        self.vocab = VocabularyConfig()
        self.ctc = CTCLossWrapper(blank=0, zero_infinity=True)

    def test_ctc_loss_finite_calculation(self) -> None:
        """Verifies CTCLossWrapper calculates a finite loss for synthetic targets."""
        batch_size = 2
        t_steps = 75
        v_size = self.vocab.vocab_size

        # Create synthetic log-probabilities [T, B, V]
        logits = torch.randn((t_steps, batch_size, v_size))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Targets [B, max_L] with lengths 5 and 4
        targets = torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]],
            dtype=torch.long,
        )
        input_lengths = torch.tensor([75, 75], dtype=torch.long)
        target_lengths = torch.tensor([5, 4], dtype=torch.long)

        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)  # Scalar mean
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_greedy_decode_collapsing(self) -> None:
        """Verifies greedy CTC decode collapses consecutive identical tokens and removes blank (0)."""
        # Encode target word dynamically
        target_ids = self.vocab.encode("bin")  # e.g. [3, 10, 15]
        b_id, i_id, n_id = target_ids

        # Construct repeated and blank-interleaved sequence
        # [blank, b, b, blank, i, n, n]
        pred_indices = [0, b_id, b_id, 0, i_id, n_id, n_id]
        v_size = self.vocab.vocab_size

        # Create one-hot logits
        logits = torch.zeros((1, len(pred_indices), v_size))
        for t, idx in enumerate(pred_indices):
            logits[0, t, idx] = 10.0

        decoded = greedy_decode(logits, blank_idx=0)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0], target_ids)

        text = decode_to_text(logits, blank_idx=0, vocab=self.vocab)
        self.assertEqual(text, ["bin"])


if __name__ == "__main__":
    unittest.main()
