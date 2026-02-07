"""Transformer Model for End-to-End Forecasting"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # d_ff defaults to 4x d_model if not specified
        dim_feedforward = d_ff if d_ff is not None else d_model * 4

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(seq_len * d_model, pred_len * output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        out = self.output_projection(x)
        return out.view(batch_size, self.pred_len, self.output_dim)


if __name__ == "__main__":
    model = TransformerModel(seq_len=168, pred_len=24)
    x = torch.randn(32, 168, 1)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
