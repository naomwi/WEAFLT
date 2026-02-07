"""
IMF Aggregator: Combines predictions from all IMF models
Uses Attention-based Weighted Sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMFAggregator(nn.Module):
    """
    Aggregates predictions from multiple IMF models using attention weights.

    Pipeline:
        [Pred_1, Pred_2, ..., Pred_n, Pred_residue] → Concat → Attention → Weighted Sum → Final

    The attention mechanism learns which IMF is more important for the final prediction.
    """

    def __init__(
        self,
        n_components: int,      # Number of IMFs + 1 (residue)
        pred_len: int,
        hidden_dim: int = 64,
        use_attention: bool = True
    ):
        super(IMFAggregator, self).__init__()
        self.n_components = n_components
        self.pred_len = pred_len
        self.use_attention = use_attention

        if use_attention:
            # Attention mechanism to weight IMF contributions
            self.attention = nn.Sequential(
                nn.Linear(n_components * pred_len, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_components),
                nn.Softmax(dim=-1)
            )
        else:
            # Learnable fixed weights
            self.weights = nn.Parameter(torch.ones(n_components) / n_components)

    def forward(self, imf_predictions: list) -> torch.Tensor:
        """
        Aggregate IMF predictions.

        Args:
            imf_predictions: List of tensors, each (batch, pred_len, 1)

        Returns:
            final_prediction: (batch, pred_len, 1)
        """
        # Stack predictions: (batch, n_components, pred_len, 1)
        stacked = torch.stack(imf_predictions, dim=1)
        batch_size = stacked.size(0)

        # Squeeze last dim: (batch, n_components, pred_len)
        stacked = stacked.squeeze(-1)

        if self.use_attention:
            # Flatten for attention input: (batch, n_components * pred_len)
            flat = stacked.reshape(batch_size, -1)

            # Compute attention weights: (batch, n_components)
            weights = self.attention(flat)

            # Weighted sum: (batch, pred_len)
            # weights: (batch, n_components) -> (batch, n_components, 1)
            # stacked: (batch, n_components, pred_len)
            weights = weights.unsqueeze(-1)
            weighted = (stacked * weights).sum(dim=1)
        else:
            # Fixed weights
            weights = F.softmax(self.weights, dim=0)
            weights = weights.view(1, -1, 1)
            weighted = (stacked * weights).sum(dim=1)

        # Add back dimension: (batch, pred_len, 1)
        return weighted.unsqueeze(-1)

    def get_attention_weights(self, imf_predictions: list) -> torch.Tensor:
        """
        Get attention weights for analysis/explainability.

        Returns:
            weights: (batch, n_components)
        """
        if not self.use_attention:
            return F.softmax(self.weights, dim=0).unsqueeze(0)

        stacked = torch.stack(imf_predictions, dim=1).squeeze(-1)
        batch_size = stacked.size(0)
        flat = stacked.reshape(batch_size, -1)
        return self.attention(flat)


class SimpleAggregator(nn.Module):
    """
    Simple sum aggregator (baseline).
    Just sums all IMF predictions without weighting.
    """

    def __init__(self, n_components: int):
        super(SimpleAggregator, self).__init__()
        self.n_components = n_components

    def forward(self, imf_predictions: list) -> torch.Tensor:
        """Simple sum of all predictions."""
        stacked = torch.stack(imf_predictions, dim=1)
        return stacked.sum(dim=1)


if __name__ == "__main__":
    print("Testing IMF Aggregator...")

    batch_size = 32
    pred_len = 24
    n_components = 13  # 12 IMFs + 1 residue

    # Create fake IMF predictions
    imf_preds = [torch.randn(batch_size, pred_len, 1) for _ in range(n_components)]

    # Test attention aggregator
    agg = IMFAggregator(n_components, pred_len, use_attention=True)
    output = agg(imf_preds)
    print(f"Attention Aggregator: output={output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in agg.parameters()):,}")

    # Get attention weights
    weights = agg.get_attention_weights(imf_preds)
    print(f"  Attention weights shape: {weights.shape}")
    print(f"  Sample weights: {weights[0].detach().numpy()}")

    # Test simple aggregator
    simple_agg = SimpleAggregator(n_components)
    output = simple_agg(imf_preds)
    print(f"Simple Aggregator: output={output.shape}")
