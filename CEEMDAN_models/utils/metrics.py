"""
Metrics Module for CEEMDAN-based Forecasting
Standard metrics: MAE, RMSE, MAPE, R²
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
    """
    Mean Absolute Percentage Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE as percentage (0-100 scale)
    """
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return 0.0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (Coefficient of Determination)

    Returns:
        R² score (1 is perfect, 0 is baseline, negative is worse than baseline)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def sudden_fluctuation_mae(y_true: np.ndarray, y_pred: np.ndarray, top_k_percent: float = 5.0) -> float:
    """
    Sudden-Fluctuation Score: MAE chỉ tính trên top-k% time steps có |Δx| lớn nhất.

    Đo khả năng dự đoán tại các điểm có biến động đột ngột.
    Model tốt phải dự đoán được cả những lúc data thay đổi nhanh.

    Args:
        y_true: Ground truth values (1D array)
        y_pred: Predicted values (1D array)
        top_k_percent: Phần trăm top fluctuations (default: 5%)

    Returns:
        MAE trên các time steps có biến động đột ngột
    """
    if len(y_true) < 2:
        return 0.0

    # Tính |Δx| = |x[t] - x[t-1]|
    delta_x = np.abs(np.diff(y_true))  # Shape: (n-1,)

    # Tìm threshold cho top k%
    threshold = np.percentile(delta_x, 100 - top_k_percent)

    # Indices của top k% fluctuations (offset +1 vì diff giảm 1 phần tử)
    sudden_indices = np.where(delta_x >= threshold)[0] + 1

    if len(sudden_indices) == 0:
        return 0.0

    # Tính MAE chỉ trên các sudden fluctuation points
    return np.mean(np.abs(y_true[sudden_indices] - y_pred[sudden_indices]))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all standard metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with all metrics
    """
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'MAE_Sudden': sudden_fluctuation_mae(y_true, y_pred, top_k_percent=5.0),
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    if prefix:
        print(f"\n{prefix}")
    print("-" * 40)
    print(f"  MAE:        {metrics['MAE']:.4f}")
    print(f"  RMSE:       {metrics['RMSE']:.4f}")
    print(f"  MAPE:       {metrics['MAPE']:.2f}%")
    print(f"  R²:         {metrics['R2']:.4f}")
    print(f"  MAE_Sudden: {metrics.get('MAE_Sudden', 0):.4f}  (top 5% fluctuations)")
    print("-" * 40)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Metrics")
