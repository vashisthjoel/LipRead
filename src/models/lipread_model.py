"""End-to-end LipRead spatiotemporal deep learning model (3D CNN + Bi-LSTM).

Architecture:
  Input: [B, T, C, H, W]
    ↓ (permute)
  [B, C, T, H, W]
    ↓ (3D CNN + spatial pooling)
  [B, T, feature_dim=96]
    ↓ (2-layer Bi-LSTM)
  [B, T, 512]
    ↓ (Linear projection)
  Logits: [B, T, vocab_size]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset_config import VocabularyConfig
from src.models.bilstm import BiLSTM
from src.models.cnn3d import CNN3D
from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.models.lipread_model")


class LipReadModel(nn.Module):
    """End-to-end sentence-level lip reading neural network.

    Combines a spatiotemporal 3D CNN front-end with a 2-layer Bidirectional LSTM
    temporal encoder and linear vocabulary projection.
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        in_channels: int = 1,
        cnn_channels: Sequence[int] = (32, 64, 96),
        lstm_hidden_size: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        ctc_blank_index: int = 0,
    ):
        """Initializes the LipReadModel architecture.

        Args:
            vocab_size: Vocabulary classification size (including CTC blank token).
                        Defaults to VocabularyConfig().vocab_size (30 classes).
            in_channels: Input image channels (default: 1 for grayscale).
            cnn_channels: Sequence of 3 integer filter counts for 3D CNN blocks.
            lstm_hidden_size: Hidden dimension per LSTM direction (default: 256).
            lstm_layers: Number of recurrent layers (default: 2).
            bidirectional: Whether to employ bidirectional recurrent processing.
            dropout: Inter-layer recurrent dropout probability (default: 0.2).
            ctc_blank_index: Index of CTC blank token (default: 0).
        """
        super().__init__()
        self.ctc_blank_index = ctc_blank_index

        # Resolve vocabulary size
        if vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            vocab_cfg = VocabularyConfig()
            self.vocab_size = vocab_cfg.vocab_size

        self.in_channels = in_channels
        self.cnn_channels = tuple(cnn_channels)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # 1. Spatiotemporal 3D CNN Front-End: [B, C, T, H, W] -> [B, T, feature_dim]
        self.frontend = CNN3D(
            in_channels=in_channels,
            channels=self.cnn_channels,
        )
        self.feature_dim = self.frontend.feature_dim

        # 2. Bidirectional LSTM Temporal Back-End: [B, T, feature_dim] -> [B, T, hidden_size * 2]
        self.backend = BiLSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        # 3. Linear Vocabulary Projection: [B, T, hidden_size * 2] -> [B, T, vocab_size]
        self.classifier = nn.Linear(self.backend.output_dim, self.vocab_size)

        logger.debug(
            "LipReadModel initialized: in_channels=%d, cnn_channels=%s, feature_dim=%d, "
            "lstm_hidden=%d, vocab_size=%d",
            in_channels,
            self.cnn_channels,
            self.feature_dim,
            lstm_hidden_size,
            self.vocab_size,
        )

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> LipReadModel:
        """Instantiates LipReadModel using parameters from central config.yaml."""
        cfg = get_config().to_dict()
        model_cfg = cfg.get("model", {})
        prep_cfg = cfg.get("preprocessing", {})

        in_channels = 1 if prep_cfg.get("grayscale", True) else 3
        cnn_channels = model_cfg.get("cnn_channels", [32, 64, 96])
        lstm_hidden_size = model_cfg.get("lstm_hidden_size", 256)
        lstm_layers = model_cfg.get("lstm_layers", 2)
        bidirectional = model_cfg.get("bidirectional", True)
        dropout = model_cfg.get("dropout", 0.2)

        return cls(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass converting raw video sequences into character class logits.

        Args:
            video: Input video tensor of shape [B, T, C, H, W].

        Returns:
            Unnormalized logits tensor of shape [B, T, vocab_size].

        Raises:
            ValueError: If input tensor does not match [B, T, C, H, W] dimension format.
        """
        if video.ndim != 5:
            raise ValueError(
                f"Expected 5D input tensor [B, T, C, H, W] for LipReadModel, but received "
                f"tensor with {video.ndim} dimensions (shape: {tuple(video.shape)})."
            )

        if video.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected channel dimension C={self.in_channels} at index 2 (shape: [B, T, C, H, W]), "
                f"but received tensor with C={video.shape[2]} (shape: {tuple(video.shape)})."
            )

        # Convert [B, T, C, H, W] -> [B, C, T, H, W] for PyTorch Conv3d
        x = video.permute(0, 2, 1, 3, 4)

        # 3D CNN Front-End: [B, C, T, H, W] -> [B, T, feature_dim]
        features = self.frontend(x)

        # Bi-LSTM Back-End: [B, T, feature_dim] -> [B, T, hidden_size * 2]
        lstm_out, _ = self.backend(features)

        # Linear projection to vocabulary logits: [B, T, vocab_size]
        logits = self.classifier(lstm_out)

        return logits

    @staticmethod
    def to_ctc_layout(logits: torch.Tensor) -> torch.Tensor:
        """Converts standard batch-first logits [B, T, V] to PyTorch CTC layout [T, B, V].

        Uses transpose to create a lightweight view without unnecessary memory copying.

        Args:
            logits: Logits or log-probabilities of shape [B, T, V].

        Returns:
            Tensor transposed to [T, B, V].
        """
        if logits.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, T, V] to transpose, got ndim={logits.ndim}")
        return logits.transpose(0, 1)

    def get_log_probs(self, video: torch.Tensor, ctc_layout: bool = False) -> torch.Tensor:
        """Convenience method returning log-softmax probabilities for CTC loss calculation.

        Args:
            video: Input tensor of shape [B, T, C, H, W].
            ctc_layout: If True, returns tensor formatted as [T, B, V]. Defaults to [B, T, V].

        Returns:
            Log-probabilities tensor of shape [B, T, vocab_size] or [T, B, vocab_size].
        """
        logits = self.forward(video)
        log_probs = F.log_softmax(logits, dim=-1)
        if ctc_layout:
            return self.to_ctc_layout(log_probs)
        return log_probs
