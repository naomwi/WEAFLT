"""
Linear Models: DLinear and NLinear
From LTSF-Linear paper, adapted for Change-aware features
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average for series decomposition."""

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad front and end
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """Series decomposition into trend and seasonal."""

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition Linear model for single IMF prediction.
    Adapted for change-aware features (5 input features).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int = 5,      # [IMF, Δx, |Δx|, rolling_std, rolling_zscore]
        output_dim: int = 1,
        kernel_size: int = 25
    ):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.decomposition = SeriesDecomp(kernel_size)

        # Per-channel linear layers
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # Project from input_dim to output_dim if needed
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - 5 change-aware features

        Returns:
            (batch, pred_len, output_dim) - prediction for this IMF
        """
        # Decompose
        seasonal_init, trend_init = self.decomposition(x)

        # Linear layers (per channel)
        seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        x = seasonal_output + trend_output

        # Project to output_dim if needed
        if self.projection is not None:
            x = self.projection(x)

        return x


class NLinear(nn.Module):
    """
    Normalization Linear model for single IMF prediction.
    Adapted for change-aware features (5 input features).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int = 5,
        output_dim: int = 1
    ):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, pred_len, output_dim)
        """
        # Normalize by last value (only for first channel - IMF value)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        # Linear
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Add back normalization
        x = x + seq_last[:, :, :self.pred_len] if seq_last.size(1) >= self.pred_len else x + seq_last

        # Project to output_dim
        if self.projection is not None:
            x = self.projection(x)

        return x


if __name__ == "__main__":
    print("Testing DLinear and NLinear with change-aware features...")

    batch_size = 32
    seq_len = 168
    pred_len = 24
    input_dim = 5   # 5 change-aware features

    # Test DLinear
    model = DLinear(seq_len, pred_len, input_dim=input_dim, output_dim=1)
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    print(f"DLinear: input={x.shape}, output={y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test NLinear
    model = NLinear(seq_len, pred_len, input_dim=input_dim, output_dim=1)
    y = model(x)
    print(f"NLinear: input={x.shape}, output={y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
