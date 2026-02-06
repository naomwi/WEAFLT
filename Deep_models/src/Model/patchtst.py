"""
PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers

Paper: https://arxiv.org/abs/2211.14730
Reference implementation: https://github.com/yuqinie98/PatchTST

Key innovations:
1. Patching: Segment time series into patches for better local semantic capture
2. Channel Independence: Process each channel independently to reduce complexity
3. Instance Normalization: RevIN (Reversible Instance Normalization) for distribution shift

This implementation is adapted for water quality forecasting with multi-target output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for handling distribution shift.

    Normalizes input and denormalizes output to handle non-stationary time series.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, features]
            mode: 'norm' for normalization, 'denorm' for denormalization
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        """Compute mean and std for normalization."""
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = x.std(dim=1, keepdim=True).detach() + self.eps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.std + self.mean
        return x


class PositionalEncoding(nn.Module):
    """Learnable or sinusoidal positional encoding for patches."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
        dropout: float = 0.1,
        learnable: bool = True
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable

        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for time series.

    Segments the input sequence into patches and projects them to d_model dimension.
    """

    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        padding_patch: str = 'end'
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # Linear projection from patch to d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Create patches from input sequence.

        Args:
            x: Input [batch, seq_len, n_vars]

        Returns:
            patches: [batch * n_vars, num_patches, d_model]
            n_vars: Number of variables
        """
        batch_size, seq_len, n_vars = x.shape

        # Padding if needed
        if self.padding_patch == 'end':
            pad_len = (self.patch_len - seq_len % self.patch_len) % self.patch_len
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
                seq_len = seq_len + pad_len

        # Number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        # Unfold to create patches: [batch, n_vars, num_patches, patch_len]
        x = x.permute(0, 2, 1)  # [batch, n_vars, seq_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Reshape for channel independence: [batch * n_vars, num_patches, patch_len]
        x = x.reshape(batch_size * n_vars, num_patches, self.patch_len)

        # Project patches: [batch * n_vars, num_patches, d_model]
        x = self.value_embedding(x)

        return x, n_vars


class PatchTSTEncoder(nn.Module):
    """
    Transformer encoder for PatchTST.

    Uses standard transformer encoder with pre-norm architecture.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * n_vars, num_patches, d_model]
        Returns:
            [batch * n_vars, num_patches, d_model]
        """
        return self.encoder(x)


class FlattenHead(nn.Module):
    """
    Prediction head that flattens patch representations and projects to output.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        num_patches: int,
        pred_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(num_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x: [batch * n_vars, num_patches, d_model]
            batch_size: Original batch size

        Returns:
            [batch, pred_len, n_vars]
        """
        # Flatten: [batch * n_vars, num_patches * d_model]
        x = self.flatten(x)
        x = self.dropout(x)

        # Project: [batch * n_vars, pred_len]
        x = self.linear(x)

        # Reshape back: [batch, n_vars, pred_len]
        x = x.reshape(batch_size, self.n_vars, self.pred_len)

        # Permute: [batch, pred_len, n_vars]
        x = x.permute(0, 2, 1)

        return x


class PatchTST(nn.Module):
    """
    PatchTST: Patch Time Series Transformer for long-term forecasting.

    Key features:
    1. Patching for local semantic extraction
    2. Channel independence for scalability
    3. Optional RevIN for distribution shift handling

    Args:
        input_dim: Number of input features/channels
        output_dim: Number of output targets
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        patch_len: Length of each patch (default: 16)
        stride: Stride between patches (default: 8)
        d_model: Transformer model dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 3)
        d_ff: Feedforward dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
        use_revin: Use Reversible Instance Normalization (default: True)
        individual: Use individual head per variable (default: False)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = True,
        individual: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_revin = use_revin

        # Calculate number of patches
        padded_seq_len = seq_len + (patch_len - seq_len % patch_len) % patch_len
        self.num_patches = (padded_seq_len - patch_len) // stride + 1

        # RevIN for distribution shift
        if use_revin:
            self.revin = RevIN(input_dim)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=self.num_patches + 10,
            dropout=dropout,
            learnable=True
        )

        # Transformer encoder
        self.encoder = PatchTSTEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        # Prediction head
        self.head = FlattenHead(
            n_vars=input_dim,
            d_model=d_model,
            num_patches=self.num_patches,
            pred_len=pred_len,
            dropout=dropout
        )

        # Output projection if input_dim != output_dim
        if input_dim != output_dim:
            self.output_projection = nn.Linear(input_dim, output_dim)
        else:
            self.output_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Output tensor [batch, pred_len, output_dim]
        """
        batch_size = x.size(0)

        # RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')

        # Patch embedding: [batch * n_vars, num_patches, d_model]
        x, n_vars = self.patch_embedding(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.encoder(x)

        # Prediction head: [batch, pred_len, n_vars]
        x = self.head(x, batch_size)

        # RevIN denormalization
        if self.use_revin:
            x = self.revin(x, mode='denorm')

        # Output projection if needed
        if self.output_projection is not None:
            x = self.output_projection(x)

        return x


class CEEMDANPatchTST(nn.Module):
    """
    CEEMDAN-PatchTST hybrid model.

    Combines CEEMDAN signal decomposition with PatchTST forecasting.
    Expects input to already contain CEEMDAN-decomposed features (IMFs + residue).

    The decomposition should be done in preprocessing, and this model
    leverages the decomposed representation for better forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        super().__init__()

        self.patchtst = PatchTST(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_revin=use_revin
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on CEEMDAN-preprocessed features.

        Args:
            x: CEEMDAN-decomposed input [batch, seq_len, features_with_imfs]

        Returns:
            Predictions [batch, pred_len, output_dim]
        """
        return self.patchtst(x)


# Alias for compatibility
PatchTSTModel = PatchTST
