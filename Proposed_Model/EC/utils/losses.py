"""
Loss Functions for Proposed Model
EventWeightedLoss: Higher weight for sudden fluctuation timesteps
"""

import torch
import torch.nn as nn
from typing import Optional


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
    print("Testing EventWeightedLoss")
    print("=" * 50)

    batch_size = 4
    pred_len = 24

    pred = torch.randn(batch_size, pred_len, 1)
    target = torch.randn(batch_size, pred_len, 1)

    # Event flags (binary)
    event_flag = torch.zeros(batch_size, pred_len)
    event_flag[:, 5] = 1.0
    event_flag[:, 15] = 1.0

    # Test EventWeightedLoss
    print("\nEventWeightedLoss (event_weight=4.0):")
    loss_fn = EventWeightedLoss(event_weight=4.0)
    loss_no_flag = loss_fn(pred, target)
    loss_with_flag = loss_fn(pred, target, event_flag)
    print(f"   Without flags: {loss_no_flag.item():.4f}")
    print(f"   With flags:    {loss_with_flag.item():.4f}")
