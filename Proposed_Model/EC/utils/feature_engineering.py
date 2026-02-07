"""
Change-Aware Feature Engineering
Features computed from ORIGINAL signal (not IMF)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def create_change_aware_features(
    data: np.ndarray,
    rolling_std_window: int = 12,
    rolling_zscore_window: int = 24
) -> Dict[str, np.ndarray]:
    """
    Create change-aware features from ORIGINAL signal.

    Features:
    - Delta_X: Δx = x[t] - x[t-1]
    - Abs_Delta_X: |Δx|
    - Rolling_Std: rolling standard deviation (12h window)
    - Rolling_Zscore: (x - rolling_mean) / rolling_std (24h window)

    Args:
        data: 1D numpy array of ORIGINAL time series (EC or pH)
        rolling_std_window: Window for rolling_std (default: 12 hours)
        rolling_zscore_window: Window for rolling_zscore (default: 24 hours)

    Returns:
        Dictionary with:
            - 'features': 2D array (n_samples, 4) [Δx, |Δx|, rolling_std, rolling_zscore]
            - 'feature_names': List of feature names
    """
    n = len(data)

    # Delta_X: first-order difference
    delta_x = np.zeros(n)
    delta_x[1:] = np.diff(data)

    # Abs_Delta_X
    abs_delta_x = np.abs(delta_x)

    # Use pandas for rolling calculations
    series = pd.Series(data)

    # Rolling Std (12h window)
    rolling_std = series.rolling(window=rolling_std_window, min_periods=1).std().values
    rolling_std = np.nan_to_num(rolling_std, nan=0.0)

    # Rolling Mean (24h window)
    rolling_mean = series.rolling(window=rolling_zscore_window, min_periods=1).mean().values

    # Rolling Std for zscore (24h window)
    rolling_std_zscore = series.rolling(window=rolling_zscore_window, min_periods=1).std().values
    rolling_std_zscore = np.nan_to_num(rolling_std_zscore, nan=1.0)
    rolling_std_zscore[rolling_std_zscore < 1e-8] = 1.0

    # Rolling Zscore
    rolling_zscore = (data - rolling_mean) / rolling_std_zscore
    rolling_zscore = np.nan_to_num(rolling_zscore, nan=0.0)

    # Stack features
    features = np.stack([
        delta_x,
        abs_delta_x,
        rolling_std,
        rolling_zscore
    ], axis=1).astype(np.float32)

    return {
        'features': features,
        'feature_names': ['Delta_X', 'Abs_Delta_X', 'Rolling_Std', 'Rolling_Zscore']
    }


def create_event_flags(
    data: np.ndarray,
    percentile: float = 95.0,
    train_end_idx: int = None
) -> Tuple[np.ndarray, float]:
    """
    Create event flags from ORIGINAL signal.

    Event = 1 if |Δx| > percentile_95 threshold
    Threshold is computed ONLY from training data.

    Args:
        data: 1D numpy array of ORIGINAL time series
        percentile: Percentile for threshold (default: 95)
        train_end_idx: Index where training data ends (for threshold computation)

    Returns:
        Tuple of (event_flags, threshold)
    """
    n = len(data)

    # Compute |Δx|
    abs_delta_x = np.zeros(n)
    abs_delta_x[1:] = np.abs(np.diff(data))

    # Compute threshold from training data only
    if train_end_idx is not None:
        train_delta = abs_delta_x[:train_end_idx]
    else:
        train_delta = abs_delta_x

    # Filter non-zero values for percentile
    nonzero = train_delta[train_delta > 1e-8]
    if len(nonzero) > 0:
        threshold = np.percentile(nonzero, percentile)
    else:
        threshold = 0.0

    # Create flags for ALL data using training threshold
    event_flags = (abs_delta_x >= threshold).astype(np.float32)

    return event_flags, threshold


def combine_imf_with_features(
    imf: np.ndarray,
    features: np.ndarray
) -> np.ndarray:
    """
    Combine IMF values with change-aware features.

    Input to each IMF model: [IMF_value, Δx, |Δx|, rolling_std, rolling_zscore]

    Args:
        imf: 1D array of IMF values (from CEEMDAN)
        features: 2D array of change-aware features (from original signal)

    Returns:
        2D array of shape (n_samples, 5)
    """
    imf = imf.reshape(-1, 1)
    return np.concatenate([imf, features], axis=1).astype(np.float32)


if __name__ == "__main__":
    print("Testing feature engineering...")

    # Generate test data with sudden changes
    np.random.seed(42)
    n = 1000
    data = np.sin(np.linspace(0, 10, n)) + 0.1 * np.random.randn(n)
    data[200] += 2.0  # Sudden change
    data[500] -= 1.5
    data[800] += 1.8

    # Test change-aware features
    result = create_change_aware_features(data, rolling_std_window=12, rolling_zscore_window=24)
    print(f"Features shape: {result['features'].shape}")
    print(f"Feature names: {result['feature_names']}")

    # Test event flags
    train_end = int(0.6 * len(data))
    event_flags, threshold = create_event_flags(data, percentile=95, train_end_idx=train_end)
    print(f"Event flags: {event_flags.sum():.0f} events")
    print(f"Threshold: {threshold:.4f}")

    # Test combining with IMF
    fake_imf = np.random.randn(n)
    combined = combine_imf_with_features(fake_imf, result['features'])
    print(f"Combined shape: {combined.shape}")  # Should be (n, 5)
