"""3D Convolutional Neural Network spatiotemporal front-end for LipRead.

Processes input video sequences [B, C, T, H, W] to extract spatiotemporal lip motion
features while strictly preserving the temporal sequence length (T=75) for CTC alignment.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn

from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.models.cnn3d")


class CNN3D(nn.Module):
    """Lightweight 3D CNN spatiotemporal feature extractor.

    Architecture:
      Block 1: Conv3d(1 -> 32, kernel=(3, 5, 5), pad=(1, 2, 2)) + BN + ReLU + MaxPool3d((1, 2, 2))
      Block 2: Conv3d(32 -> 64, kernel=(3, 3, 3), pad=(1, 1, 1)) + BN + ReLU + MaxPool3d((1, 2, 2))
      Block 3: Conv3d(64 -> 96, kernel=(3, 3, 3), pad=(1, 1, 1)) + BN + ReLU + MaxPool3d((1, 2, 2))
      Spatial Aggregation: AdaptiveAvgPool3d((None, 1, 1)) -> [B, T, feature_dim]

    Spatial dimension reduction:
      88x88 -> 44x44 -> 22x22 -> 11x11 -> 1x1 (spatial pooling)
    Temporal dimension:
      75 -> 75 -> 75 -> 75 -> 75 (fully preserved)
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 96),
    ):
        """Initializes 3D CNN layers.

        Args:
            in_channels: Number of input frame channels (1 for grayscale).
            channels: Feature channel dimensions for blocks 1, 2, and 3.
        """
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"channels must contain exactly 3 integers, got {channels}")

        c1, c2, c3 = channels
        self.in_channels = in_channels
        self.out_channels = c3

        # Block 1: Captures initial motion and local spatial patterns
        self.block1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=c1,
                kernel_size=(3, 5, 5),
                stride=(1, 1, 1),
                padding=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Block 2: Intermediate spatiotemporal representation
        self.block2 = nn.Sequential(
            nn.Conv3d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Block 3: Higher-level viseme motion dynamics
        self.block3 = nn.Sequential(
            nn.Conv3d(
                in_channels=c2,
                out_channels=c3,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Global spatial pooling: collapses (H, W) to (1, 1) while preserving temporal T
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    @property
    def feature_dim(self) -> int:
        """Returns the output feature dimension per timestep."""
        return self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D CNN.

        Args:
            x: Input tensor of shape [B, C, T, H, W].

        Returns:
            Temporal feature sequence tensor of shape [B, T, feature_dim].

        Raises:
            ValueError: If input tensor does not have 5 dimensions or channel mismatch.
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D tensor [B, C, T, H, W] for 3D CNN, but received tensor with "
                f"{x.ndim} dimensions (shape: {tuple(x.shape)})."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected channel dimension C={self.in_channels}, but received tensor with "
                f"C={x.shape[1]} (shape: {tuple(x.shape)})."
            )

        # Spatiotemporal convolution blocks
        # [B, C, T, H, W] -> [B, c1, T, H/2, W/2]
        x = self.block1(x)

        # [B, c1, T, H/2, W/2] -> [B, c2, T, H/4, W/4]
        x = self.block2(x)

        # [B, c2, T, H/4, W/4] -> [B, c3, T, H/8, W/8]
        x = self.block3(x)

        # Spatial pooling: [B, c3, T, H/8, W/8] -> [B, c3, T, 1, 1]
        x = self.spatial_pool(x)

        # Squeeze spatial dimensions: [B, c3, T, 1, 1] -> [B, c3, T]
        x = x.squeeze(-1).squeeze(-1)

        # Transpose channel and temporal dimensions: [B, c3, T] -> [B, T, c3]
        x = x.transpose(1, 2)

        return x
