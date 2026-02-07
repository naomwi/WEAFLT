"""PatchTST Model for End-to-End Forecasting"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        num_patches = x.size(2)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, num_patches, -1)
        return self.projection(x)


class PatchTST(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int = 1,
        output_dim: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embedding = PatchEmbedding(input_dim, patch_len, stride, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 10, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * d_model, pred_len * output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        out = self.head(x)
        return out.view(batch_size, self.pred_len, self.output_dim)


if __name__ == "__main__":
    model = PatchTST(seq_len=168, pred_len=24)
    x = torch.randn(32, 168, 1)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
