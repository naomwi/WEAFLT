"""LSTM Model for End-to-End Forecasting"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Adjust fc input size for bidirectional
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, pred_len * output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.view(batch_size, self.pred_len, self.output_dim)


if __name__ == "__main__":
    model = LSTMModel(seq_len=168, pred_len=24)
    x = torch.randn(32, 168, 1)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
