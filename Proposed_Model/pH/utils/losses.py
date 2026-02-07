"""
Loss Functions for Proposed Model
EventWeightedLoss: Higher weight for sudden fluctuation timesteps
AdaptiveWeightedLoss: Weight proportional to deviation magnitude
"""

import torch
import torch.nn as nn
from typing import Optional


class AdaptiveWeightedLoss(nn.Module):
    """
    Adaptive Weighted Loss Function.

    Weight is proportional to the magnitude of deviation (|Δx|).
    Larger deviations (outliers) get higher weights automatically.

    Weight formula: w(t) = 1 + alpha * (|Δx(t)| / mean(|Δx|))

    Benefits:
    - Continuous weighting (not binary like EventWeightedLoss)
    - Works for both upward and downward outliers
    - Self-adaptive: no need to tune fixed event_weight

    Args:
        alpha: Scaling factor for adaptive weight (default: 1.0)
        min_weight: Minimum weight (default: 1.0)
        max_weight: Maximum weight cap (default: 10.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_weight: float = 1.0,
        max_weight: float = 10.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        abs_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute adaptive weighted MSE loss.

        Args:
            pred: Predicted values (batch, pred_len, dim) or (batch, pred_len)
            target: Target values, same shape as pred
            abs_delta: |Δx| values for each timestep (batch, pred_len)
                       If None, uses uniform weighting (standard MSE)

        Returns:
            Loss value
        """
        # Compute MSE per element
        mse = (pred - target) ** 2

        if abs_delta is None:
            # Standard MSE
            if self.reduction == 'mean':
                return mse.mean()
            elif self.reduction == 'sum':
                return mse.sum()
            else:
                return mse

        # Ensure abs_delta has correct shape
        if abs_delta.dim() < mse.dim():
            abs_delta = abs_delta.unsqueeze(-1)

        # Compute mean |Δx| for normalization (avoid division by zero)
        mean_delta = abs_delta.mean() + 1e-8

        # Compute adaptive weights: w = 1 + alpha * (|Δx| / mean(|Δx|))
        weights = self.min_weight + self.alpha * (abs_delta / mean_delta)

        # Clamp weights to [min_weight, max_weight]
        weights = torch.clamp(weights, self.min_weight, self.max_weight)

        # Apply weights
        weighted_mse = mse * weights

        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


class EventWeightedLoss(nn.Module):
    """
    Event-Weighted Loss Function.

    Applies higher weight to timesteps with sudden fluctuations (events).

    Loss = mean(w(t) * MSE(t))

    Where:
        w(t) = 1.0 if non-event
        w(t) = event_weight if event

    Event flag is computed from ORIGINAL signal (not IMF).
    """

    def __init__(self, event_weight: float = 3.0, reduction: str = 'mean'):
        """
        Args:
            event_weight: Weight multiplier for event timesteps (default: 3.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.event_weight = event_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        event_flag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            pred: Predicted values (batch, pred_len, dim) or (batch, pred_len)
            target: Target values, same shape as pred
            event_flag: Binary flags (batch, pred_len) indicating events

        Returns:
            Loss value
        """
        # Compute MSE per element
        mse = (pred - target) ** 2

        if event_flag is None:
            # Standard MSE
            if self.reduction == 'mean':
                return mse.mean()
            elif self.reduction == 'sum':
                return mse.sum()
            else:
                return mse

        # Ensure event_flag has correct shape
        if event_flag.dim() < mse.dim():
            event_flag = event_flag.unsqueeze(-1)

        # Compute weights: 1 + event_flag * (event_weight - 1)
        weights = 1.0 + event_flag * (self.event_weight - 1.0)

        # Apply weights
        weighted_mse = mse * weights

        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Loss Functions")
    print("=" * 50)

    batch_size = 4
    pred_len = 24

    pred = torch.randn(batch_size, pred_len, 1)
    target = torch.randn(batch_size, pred_len, 1)

    # Event flags (binary)
    event_flag = torch.zeros(batch_size, pred_len)
    event_flag[:, 5] = 1.0
    event_flag[:, 15] = 1.0

    # |Δx| values (continuous) - simulate some outliers
    abs_delta = torch.abs(torch.randn(batch_size, pred_len)) * 0.5
    abs_delta[:, 5] = 5.0   # Large outlier
    abs_delta[:, 15] = 3.0  # Medium outlier

    # Test EventWeightedLoss
    print("\n1. EventWeightedLoss (event_weight=4.0):")
    loss_fn = EventWeightedLoss(event_weight=4.0)
    loss_no_flag = loss_fn(pred, target)
    loss_with_flag = loss_fn(pred, target, event_flag)
    print(f"   Without flags: {loss_no_flag.item():.4f}")
    print(f"   With flags:    {loss_with_flag.item():.4f}")

    # Test AdaptiveWeightedLoss
    print("\n2. AdaptiveWeightedLoss (alpha=1.0):")
    adaptive_loss_fn = AdaptiveWeightedLoss(alpha=1.0, max_weight=10.0)
    loss_no_delta = adaptive_loss_fn(pred, target)
    loss_with_delta = adaptive_loss_fn(pred, target, abs_delta)
    print(f"   Without abs_delta: {loss_no_delta.item():.4f}")
    print(f"   With abs_delta:    {loss_with_delta.item():.4f}")

    # Show weight distribution
    mean_delta = abs_delta.mean() + 1e-8
    weights = 1.0 + 1.0 * (abs_delta / mean_delta)
    weights = torch.clamp(weights, 1.0, 10.0)
    print(f"\n   Weight stats:")
    print(f"   - Min weight:  {weights.min().item():.2f}")
    print(f"   - Max weight:  {weights.max().item():.2f}")
    print(f"   - Mean weight: {weights.mean().item():.2f}")
    print(f"   - Weight at outlier (idx=5):  {weights[0, 5].item():.2f}")
    print(f"   - Weight at normal (idx=10): {weights[0, 10].item():.2f}")
