"""
LSTM Model for CEEMDAN-based Forecasting
Simple LSTM implementation for single IMF/residue prediction
"""

import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_CONFIG


class LSTMModel(nn.Module):
    """
    Simple LSTM model for sequence-to-sequence prediction.
    Predicts single component (IMF or residue).
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        seq_len: int = 168,
        pred_len: int = 24,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None
    ):
        super().__init__()

        # Use config defaults if not specified
        cfg = MODEL_CONFIG['lstm']
        self.hidden_size = hidden_size or cfg['hidden_size']
        self.num_layers = num_layers or cfg['num_layers']
        self.dropout = dropout or cfg['dropout']

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False
        )

        # Output projection
        self.fc = nn.Linear(self.hidden_size, pred_len * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, pred_len, output_dim)
        """
        batch_size = x.size(0)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Project to output
        out = self.fc(last_hidden)  # (batch, pred_len * output_dim)

        # Reshape to (batch, pred_len, output_dim)
        out = out.view(batch_size, self.pred_len, self.output_dim)

        return out


if __name__ == "__main__":
    # Test model
    print("Testing LSTM model...")

    model = LSTMModel(
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
