"""Bidirectional LSTM temporal encoder for LipRead.

Processes temporal feature vectors [B, T, feature_dim] to capture contextual bidirectional
dependencies across visual speech frames.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.models.bilstm")


class BiLSTM(nn.Module):
    """2-layer Bidirectional Long Short-Term Memory network.

    Receives temporal feature sequence [B, T, input_size] and produces bidirectional
    context representations [B, T, hidden_size * 2].
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        """Initializes the Bidirectional LSTM encoder.

        Args:
            input_size: Input feature dimension per timestep (e.g. 96 from CNN3D).
            hidden_size: Hidden state dimension per direction (default: 256).
            num_layers: Number of recurrent layers (default: 2).
            bidirectional: Whether to use bidirectional processing (default: True).
            dropout: Dropout probability between recurrent layers (default: 0.2).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_size * self.num_directions

        # PyTorch only applies dropout if num_layers > 1
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=recurrent_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through Bi-LSTM.

        Args:
            x: Input tensor of shape [B, T, input_size].
            hx: Optional tuple of initial hidden states (h_0, c_0).

        Returns:
            Tuple of:
              - Output tensor of shape [B, T, hidden_size * num_directions] (e.g. [B, T, 512]).
              - Tuple of final hidden states (h_n, c_n).

        Raises:
            ValueError: If input tensor does not have 3 dimensions or feature size mismatch.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor [B, T, input_size] for BiLSTM, but received tensor with "
                f"{x.ndim} dimensions (shape: {tuple(x.shape)})."
            )

        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected input feature dimension {self.input_size}, but received tensor with "
                f"dim={x.shape[2]} (shape: {tuple(x.shape)})."
            )

        out, (hn, cn) = self.lstm(x, hx)
        return out, (hn, cn)
