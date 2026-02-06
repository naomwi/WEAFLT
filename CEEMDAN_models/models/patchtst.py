"""
PatchTST Model for CEEMDAN-based Forecasting
Patch-based Time Series Transformer for single IMF/residue prediction
"""

import torch
import torch.nn as nn
import math
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_CONFIG


class PatchEmbedding(nn.Module):
    """Convert time series into patches."""

    def __init__(self, input_dim: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Linear projection for each patch
        self.projection = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            patches: (batch, num_patches, d_model)
        """
        batch_size, seq_len, input_dim = x.shape

        # Unfold to create patches
        # (batch, input_dim, seq_len) -> unfold -> (batch, input_dim, num_patches, patch_len)
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # (batch, input_dim, num_patches, patch_len) -> (batch, num_patches, input_dim * patch_len)
        num_patches = x.size(2)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, num_patches, -1)

        # Project to d_model
        x = self.projection(x)  # (batch, num_patches, d_model)

        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""

    def __init__(self, d_model: int, max_patches: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchTST(nn.Module):
    """
    Patch-based Time Series Transformer.
    Simple implementation for single component prediction.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        seq_len: int = 168,
        pred_len: int = 24,
        patch_len: int = None,
        stride: int = None,
        d_model: int = None,
        nhead: int = None,
        num_layers: int = None,
        dropout: float = None
    ):
        super().__init__()

        # Use config defaults if not specified
        cfg = MODEL_CONFIG['patchtst']
        self.patch_len = patch_len or cfg['patch_len']
        self.stride = stride or cfg['stride']
        self.d_model = d_model or cfg['d_model']
        self.nhead = nhead or cfg['nhead']
        self.num_layers = num_layers or cfg['num_layers']
        self.dropout = dropout or cfg['dropout']

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Calculate number of patches
        self.num_patches = (seq_len - self.patch_len) // self.stride + 1

        # Patch embedding
        self.patch_embedding = PatchEmbedding(input_dim, self.patch_len, self.stride, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_patches=self.num_patches + 10, dropout=self.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * self.d_model, pred_len * output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, pred_len, output_dim)
        """
        batch_size = x.size(0)

        # Create patches
        x = self.patch_embedding(x)  # (batch, num_patches, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, num_patches, d_model)

        # Output projection
        out = self.head(x)  # (batch, pred_len * output_dim)

        # Reshape to (batch, pred_len, output_dim)
        out = out.view(batch_size, self.pred_len, self.output_dim)

        return out


if __name__ == "__main__":
    # Test model
    print("Testing PatchTST model...")

    model = PatchTST(
        input_dim=1,
        output_dim=1,
        seq_len=168,
        pred_len=24
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of patches: {model.num_patches}")

    # Test forward pass
    x = torch.randn(32, 168, 1)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
