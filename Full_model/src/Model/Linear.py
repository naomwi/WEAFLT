import torch 
import torch.nn as nn

class DLinear(nn.Module):
    def __init__(self,input_dim,output_dim,seq_len,pred_len,kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.avg = nn.AvgPool1d(kernel_size=kernel_size,stride=1,padding=0)
        self.kernel_size = kernel_size

        self.linear_trend = nn.Linear(seq_len,pred_len)
        self.linear_seasonal = nn.Linear(seq_len,pred_len)

        self.feature_project = nn.Linear(input_dim,output_dim)

    def decompose(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal

    def forward(self, x):
        
        trend, seasonal = self.decompose(x)
        
        trend = trend.permute(0, 2, 1)         # [Batch, 54, 96]
        seasonal = seasonal.permute(0, 2, 1)   # [Batch, 54, 96]
        
        trend_pred = self.linear_trend(trend)        # -> [Batch, 54, 24]
        seasonal_pred = self.linear_seasonal(seasonal) # -> [Batch, 54, 24]
        
        pred = (trend_pred + seasonal_pred).permute(0, 2, 1)
        
        pred = self.feature_project(pred)
        
        return pred

class LTSF_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, pred_len, dropout=0.05):
        super().__init__()
        self.time_linear = nn.Linear(seq_len, pred_len)
        self.feature_linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.time_linear(x.permute(0,2,1)).permute(0,2,1)
        x = self.feature_linear(self.dropout(self.relu(x)))
        return x

class NLinear(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, pred_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Dự báo thời gian
        self.linear = nn.Linear(seq_len, pred_len)
        
        self.feature_projection = nn.Linear(input_dim, output_dim)
        self.last_val_projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        last_value = x[:, -1:, :] # [Batch, 1, Input_Dim]
        x_norm = x - last_value
        
        x_norm = x_norm.permute(0, 2, 1)     # [Batch, In, Seq]
        pred_norm = self.linear(x_norm)      # [Batch, In, Pred]
        pred_norm = pred_norm.permute(0, 2, 1) # [Batch, Pred, In]
        
        pred_norm_projected = self.feature_projection(pred_norm)
        
        last_value_projected = self.last_val_projection(last_value)
        
        pred = pred_norm_projected + last_value_projected

        return pred


class RLinear(nn.Module):
    """
    RLinear: Residual Linear model for time series forecasting.

    A simple yet effective linear model that learns residual patterns.
    Uses RevIN (Reversible Instance Normalization) for better handling
    of distribution shift in non-stationary time series.

    Reference: "Are Transformers Effective for Time Series Forecasting?"

    Args:
        input_dim: Number of input features
        output_dim: Number of output targets
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        individual: If True, use separate linear for each channel
        rev_in: If True, apply reversible instance normalization
    """

    def __init__(self, input_dim, output_dim, seq_len, pred_len, individual=False, rev_in=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.individual = individual
        self.rev_in = rev_in

        # RevIN parameters
        if self.rev_in:
            self.affine_weight = nn.Parameter(torch.ones(input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(input_dim))

        # Linear layers
        if self.individual:
            # Separate linear layer per channel
            self.linears = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(input_dim)
            ])
        else:
            # Shared linear layer across channels
            self.linear = nn.Linear(seq_len, pred_len)

        # Feature projection (input_dim -> output_dim)
        self.feature_projection = nn.Linear(input_dim, output_dim)

    def _rev_in_norm(self, x):
        """Apply reversible instance normalization."""
        # x: [batch, seq_len, input_dim]
        self._mean = x.mean(dim=1, keepdim=True)  # [batch, 1, input_dim]
        self._std = x.std(dim=1, keepdim=True) + 1e-5  # [batch, 1, input_dim]
        x = (x - self._mean) / self._std
        x = x * self.affine_weight + self.affine_bias
        return x

    def _rev_in_denorm(self, x):
        """Reverse the instance normalization."""
        # x: [batch, pred_len, input_dim]
        x = (x - self.affine_bias) / (self.affine_weight + 1e-5)
        x = x * self._std + self._mean
        return x

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Output tensor [batch, pred_len, output_dim]
        """
        # Apply RevIN normalization
        if self.rev_in:
            x = self._rev_in_norm(x)

        # Temporal projection
        if self.individual:
            # Process each channel separately
            x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
            outputs = []
            for i, linear in enumerate(self.linears):
                outputs.append(linear(x[:, i, :]))  # [batch, pred_len]
            x = torch.stack(outputs, dim=2)  # [batch, pred_len, input_dim]
        else:
            # Shared processing
            x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
            x = self.linear(x)  # [batch, input_dim, pred_len]
            x = x.permute(0, 2, 1)  # [batch, pred_len, input_dim]

        # Apply RevIN denormalization
        if self.rev_in:
            x = self._rev_in_denorm(x)

        # Feature projection
        x = self.feature_projection(x)  # [batch, pred_len, output_dim]

        return x
