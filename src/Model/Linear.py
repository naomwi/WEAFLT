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
