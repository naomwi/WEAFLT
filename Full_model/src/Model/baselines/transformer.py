"""
Transformer baseline model for water quality time series forecasting.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer-based model for multivariate time series forecasting.

    Architecture:
    - Input projection to d_model dimension
    - Positional encoding
    - Transformer encoder layers
    - Temporal and feature projection heads

    Args:
        input_dim: Number of input features
        output_dim: Number of output targets
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        d_model: Transformer model dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 2)
        dim_feedforward: Feedforward network dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(seq_len, pred_len) + 100, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Temporal projection: seq_len -> pred_len
        self.temporal_projection = nn.Linear(seq_len, pred_len)

        # Output projection: d_model -> output_dim
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]

        Returns:
            Output tensor of shape [batch, pred_len, output_dim]
        """
        # Input projection: [batch, seq_len, input_dim] -> [batch, seq_len, d_model]
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        enc_output = self.transformer_encoder(x)  # [batch, seq_len, d_model]

        # Temporal projection: [batch, d_model, seq_len] -> [batch, d_model, pred_len]
        enc_output = enc_output.permute(0, 2, 1)  # [batch, d_model, seq_len]
        temporal_out = self.temporal_projection(enc_output)  # [batch, d_model, pred_len]
        temporal_out = temporal_out.permute(0, 2, 1)  # [batch, pred_len, d_model]

        # Output projection: [batch, pred_len, d_model] -> [batch, pred_len, output_dim]
        output = self.output_projection(temporal_out)

        return output


class CEEMDANTransformerModel(nn.Module):
    """
    CEEMDAN-Transformer hybrid model.

    Uses Transformer on pre-decomposed CEEMDAN features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.transformer = TransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on CEEMDAN-preprocessed input.
        """
        return self.transformer(x)
