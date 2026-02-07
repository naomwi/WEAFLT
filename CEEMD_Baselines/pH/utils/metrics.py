"""Metrics Module"""
import numpy as np
from typing import Dict


def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def rmse(y_true, y_pred): return np.sqrt(mse(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-8):
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def sudden_fluctuation_mae(y_true, y_pred, top_k_percent=5.0):
    if len(y_true) < 2: return 0.0
    delta_x = np.abs(np.diff(y_true))
    threshold = np.percentile(delta_x, 100 - top_k_percent)
    sudden_idx = np.where(delta_x >= threshold)[0] + 1
    return np.mean(np.abs(y_true[sudden_idx] - y_pred[sudden_idx])) if len(sudden_idx) > 0 else 0.0

def calculate_all_metrics(y_true, y_pred):
    return {
        'MAE': mae(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAE_Sudden': sudden_fluctuation_mae(y_true, y_pred),
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty print metrics in formatted table."""
    if prefix:
        print(f"\n{prefix}")
    print("-" * 50)
    print(f"  MAE:        {metrics['MAE']:.4f}")
    print(f"  MSE:        {metrics.get('MSE', metrics['MAE']**2):.4f}")
    print(f"  RMSE:       {metrics['RMSE']:.4f}")
    print(f"  MAPE:       {metrics['MAPE']:.2f}%")
    print(f"  R²:         {metrics['R2']:.4f}")
    print(f"  MAE_Sudden: {metrics['MAE_Sudden']:.4f}")
    print("-" * 50)
