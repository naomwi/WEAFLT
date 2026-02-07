"""
Metrics Module
"""

import numpy as np
from typing import Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² Score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def sudden_fluctuation_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k_percent: float = 5.0
) -> float:
    """
    MAE on top-k% sudden fluctuation timesteps.
    """
    if len(y_true) < 2:
        return 0.0

    delta_x = np.abs(np.diff(y_true))
    threshold = np.percentile(delta_x, 100 - top_k_percent)
    sudden_indices = np.where(delta_x >= threshold)[0] + 1

    if len(sudden_indices) == 0:
        return 0.0

    return np.mean(np.abs(y_true[sudden_indices] - y_pred[sudden_indices]))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all metrics."""
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'MAE_Sudden': sudden_fluctuation_mae(y_true, y_pred),
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics."""
    if prefix:
        print(f"\n{prefix}")
    print("-" * 50)
    print(f"  MAE:        {metrics['MAE']:.4f}")
    print(f"  RMSE:       {metrics['RMSE']:.4f}")
    print(f"  MAPE:       {metrics['MAPE']:.2f}%")
    print(f"  R²:         {metrics['R2']:.4f}")
    print(f"  MAE_Sudden: {metrics['MAE_Sudden']:.4f}")
    print("-" * 50)
