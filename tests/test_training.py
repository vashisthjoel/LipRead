"""Unit tests for Stage 5 Training Pipeline in LipRead.

Verifies:
- Seed reproducibility across CPU/CUDA/NumPy
- Optimizer and scheduler creation
- CER and WER calculation with edge cases
- Checkpoint saving and loading
- Single training step on synthetic tensors
- Gradient accumulation mechanics
- Weight update verification after backprop + optimizer step
- Finiteness of CTC loss under synthetic forward/backward
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset_config import VocabularyConfig
from src.models.ctc import CTCLossWrapper
from src.models.lipread_model import LipReadModel
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.metrics import (
    compute_batch_cer,
    compute_batch_wer,
    compute_cer,
    compute_edit_distance,
    compute_wer,
)
from src.training.scheduler import build_optimizer, build_scheduler, get_current_lr
from src.training.trainer import Trainer
from src.utils.seed import set_seed


class SyntheticLipDataset(Dataset):
    """Synthetic dataset yielding tiny [T=75, C=1, H=88, W=88] sequences for unit tests."""

    def __init__(self, size: int = 4, seq_len: int = 75):
        self.size = size
        self.seq_len = seq_len
        self.vocab = VocabularyConfig()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        frames = torch.randn(self.seq_len, 1, 88, 88, dtype=torch.float32)
        transcript = "bin blue at f two now"
        tokens = torch.tensor(self.vocab.encode(transcript), dtype=torch.long)
        return {
            "frames": frames,
            "tokens": tokens,
            "transcript": transcript,
            "sample_id": f"synthetic_{idx}",
        }


def synthetic_collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    batch_size = len(batch)
    input_lengths = torch.tensor([b["frames"].shape[0] for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([b["tokens"].numel() for b in batch], dtype=torch.long)
    max_len = int(target_lengths.max().item())

    tokens = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        tokens[i, : b["tokens"].numel()] = b["tokens"]

    return {
        "frames": frames,
        "tokens": tokens,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "transcripts": [b["transcript"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


class TestTrainingComponents(unittest.TestCase):
    """Unit tests for training utilities, metrics, checkpointing, and scheduler."""

    def setUp(self) -> None:
        self.vocab = VocabularyConfig()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_seed_reproducibility(self) -> None:
        """Verifies random seeds generate deterministic numbers across runs."""
        set_seed(1234)
        r1 = torch.randn(5, 5)
        np1 = np.random.rand(5)

        set_seed(1234)
        r2 = torch.randn(5, 5)
        np2 = np.random.rand(5)

        self.assertTrue(torch.allclose(r1, r2))
        self.assertTrue(np.allclose(np1, np2))

    def test_edit_distance_and_cer(self) -> None:
        """Verifies Levenshtein distance and character error rate edge cases."""
        # Perfect match
        self.assertEqual(compute_edit_distance("hello", "hello"), 0)
        self.assertEqual(compute_cer("hello", "hello"), 0.0)

        # Single substitution ('e' -> 'a')
        self.assertEqual(compute_edit_distance("cat", "cot"), 1)
        self.assertAlmostEqual(compute_cer("cat", "cot"), 1 / 3)

        # Empty strings
        self.assertEqual(compute_cer("", ""), 0.0)
        self.assertEqual(compute_cer("", "abc"), 1.0)
        self.assertEqual(compute_cer("abc", ""), 1.0)

    def test_wer_calculation(self) -> None:
        """Verifies word error rate calculations."""
        # Identical
        self.assertEqual(compute_wer("bin blue at f two", "bin blue at f two"), 0.0)

        # 1 insertion
        self.assertAlmostEqual(compute_wer("bin blue", "bin blue now"), 1 / 2)

        # Batch WER
        refs = ["bin blue", "lay green"]
        hyps = ["bin red", "lay green"]
        self.assertAlmostEqual(compute_batch_wer(refs, hyps), 1 / 4)

    def test_optimizer_and_scheduler_creation(self) -> None:
        """Verifies optimizer and ReduceLROnPlateau creation and learning rate readout."""
        model = nn.Linear(10, 2)
        optimizer = build_optimizer(model, lr=0.001, weight_decay=1e-4)
        self.assertAlmostEqual(get_current_lr(optimizer), 0.001)

        scheduler = build_scheduler(optimizer, scheduler_type="plateau", patience=1)
        self.assertIsNotNone(scheduler)

    def test_checkpoint_save_and_load(self) -> None:
        """Verifies full checkpoint save and restore cycle."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        state = {
            "epoch": 3,
            "global_step": 150,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": 1.234,
            "random_seed": 42,
        }

        latest_path, best_path = save_checkpoint(
            state=state,
            is_best=True,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.assertTrue(latest_path.exists())
        self.assertIsNotNone(best_path)
        self.assertTrue(best_path.exists())

        # Create new model and load state
        new_model = nn.Linear(10, 2)
        new_opt = torch.optim.Adam(new_model.parameters(), lr=0.001)
        meta = load_checkpoint(latest_path, model=new_model, optimizer=new_opt, device="cpu")

        self.assertEqual(meta["epoch"], 3)
        self.assertEqual(meta["global_step"], 150)
        self.assertAlmostEqual(meta["best_validation_loss"], 1.234)

        # Check parameter equality
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))


class TestTrainerExecution(unittest.TestCase):
    """Tests forward, backward, CTC loss, gradient accumulation, and weight updates."""

    def setUp(self) -> None:
        self.vocab = VocabularyConfig()
        self.model = LipReadModel(
            vocab_size=self.vocab.vocab_size,
            cnn_channels=[16, 32, 48],  # Tiny CNN for fast test execution
            lstm_hidden_size=64,
            lstm_layers=1,
            bidirectional=True,
            ctc_blank_index=0,
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_single_training_step_and_weight_update(self) -> None:
        """Verifies that a training step executes forward, backward, and updates parameters."""
        dataset = SyntheticLipDataset(size=2, seq_len=75)
        loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate_fn)

        optimizer = build_optimizer(self.model, lr=0.01)

        # Capture initial weight copy
        first_param_before = next(self.model.parameters()).clone()

        trainer = Trainer(
            model=self.model,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer,
            vocab=self.vocab,
            device="cpu",
            use_amp=False,
            gradient_accumulation_steps=1,
            checkpoint_dir=self.checkpoint_dir,
        )

        stats = trainer.train_epoch(epoch=1)

        self.assertIn("train_loss", stats)
        self.assertTrue(stats["train_loss"] > 0)
        self.assertTrue(np.isfinite(stats["train_loss"]))

        # Verify weights have changed
        first_param_after = next(self.model.parameters())
        self.assertFalse(torch.allclose(first_param_before, first_param_after))

    def test_gradient_accumulation(self) -> None:
        """Verifies trainer handles gradient accumulation steps cleanly."""
        dataset = SyntheticLipDataset(size=4, seq_len=75)
        loader = DataLoader(dataset, batch_size=1, collate_fn=synthetic_collate_fn)
        optimizer = build_optimizer(self.model, lr=0.001)

        trainer = Trainer(
            model=self.model,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer,
            vocab=self.vocab,
            device="cpu",
            use_amp=False,
            gradient_accumulation_steps=2,
            checkpoint_dir=self.checkpoint_dir,
        )

        stats = trainer.train_epoch(epoch=1)
        self.assertEqual(stats["batches_processed"], 4)
        self.assertTrue(np.isfinite(stats["train_loss"]))


if __name__ == "__main__":
    unittest.main()
