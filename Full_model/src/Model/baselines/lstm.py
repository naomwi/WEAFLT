"""
LSTM baseline model for water quality time series forecasting.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based model for multivariate time series forecasting.

    Architecture:
    - Multi-layer LSTM encoder
    - Linear projection to prediction horizon
    - Feature projection for multi-target output

    Args:
        input_dim: Number of input features
        output_dim: Number of output targets
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        hidden_size: LSTM hidden state dimension (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        bidirectional: Use bidirectional LSTM (default: False)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Direction multiplier for bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Temporal projection: seq_len -> pred_len
        lstm_output_size = hidden_size * self.num_directions
        self.temporal_fc = nn.Linear(seq_len, pred_len)

        # Feature projection: lstm_output_size -> output_dim
        self.feature_fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]

        Returns:
            Output tensor of shape [batch, pred_len, output_dim]
        """
        batch_size = x.size(0)

        # LSTM encoding: [batch, seq_len, input_dim] -> [batch, seq_len, hidden*directions]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Temporal projection: [batch, hidden*dir, seq_len] -> [batch, hidden*dir, pred_len]
        lstm_out = lstm_out.permute(0, 2, 1)  # [batch, hidden*dir, seq_len]
        temporal_out = self.temporal_fc(lstm_out)  # [batch, hidden*dir, pred_len]
        temporal_out = temporal_out.permute(0, 2, 1)  # [batch, pred_len, hidden*dir]

        # Feature projection: [batch, pred_len, hidden*dir] -> [batch, pred_len, output_dim]
        output = self.feature_fc(temporal_out)

        return output


class CEEMDANLSTMModel(nn.Module):
    """
    CEEMDAN-LSTM hybrid model.

    Processes IMF components separately then combines predictions.
    Designed for use with pre-decomposed CEEMDAN features.

    Note: This wrapper expects the input to already contain CEEMDAN-decomposed features.
    The decomposition should be done in the data preprocessing pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Use standard LSTM on CEEMDAN-processed features
        self.lstm = LSTMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on CEEMDAN-preprocessed input.

        Args:
            x: CEEMDAN-decomposed input [batch, seq_len, features_with_imfs]

        Returns:
            Predictions [batch, pred_len, output_dim]
        """
        return self.lstm(x)
