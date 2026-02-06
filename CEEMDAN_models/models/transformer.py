"""
Transformer Model for CEEMDAN-based Forecasting
Simple Transformer encoder implementation for single IMF/residue prediction
"""

import torch
import torch.nn as nn
import math
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Simple Transformer encoder model for sequence-to-sequence prediction.
    Predicts single component (IMF or residue).
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        seq_len: int = 168,
        pred_len: int = 24,
        d_model: int = None,
        nhead: int = None,
        num_layers: int = None,
        dropout: float = None
    ):
        super().__init__()

        # Use config defaults if not specified
        cfg = MODEL_CONFIG['transformer']
        self.d_model = d_model or cfg['d_model']
        self.nhead = nhead or cfg['nhead']
        self.num_layers = num_layers or cfg['num_layers']
        self.dropout = dropout or cfg['dropout']

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=seq_len + 100, dropout=self.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output projection
        self.fc_out = nn.Linear(self.d_model * seq_len, pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, pred_len, output_dim)
        """
        batch_size = x.size(0)

        # Embed input
        x = self.input_embedding(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Flatten and project to output
        x = x.reshape(batch_size, -1)  # (batch, seq_len * d_model)
        out = self.fc_out(x)  # (batch, pred_len * output_dim)

        # Reshape to (batch, pred_len, output_dim)
        out = out.view(batch_size, self.pred_len, self.output_dim)

        return out


if __name__ == "__main__":
    # Test model
    print("Testing Transformer model...")

    model = TransformerModel(
        input_dim=1,
        output_dim=1,
        seq_len=168,
        pred_len=24
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(32, 168, 1)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
