"""
Data Loader for Proposed Model
Handles IMF data with change-aware features and event flags
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
from typing import Tuple, Dict, Optional, List


def load_raw_data(data_path: str, target_col: str, site_no: int = 1463500) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load raw data from CSV file and filter by station.

    Args:
        data_path: Path to CSV file
        target_col: Target column name (e.g., 'EC', 'pH')
        site_no: USGS site number to filter (default: 1463500, same as CEEMDAN_models)

    Returns:
        Tuple of (DataFrame, target array)
    """
    df = pd.read_csv(data_path)

    # Filter by site_no (same as CEEMDAN_models)
    if 'site_no' in df.columns and site_no is not None:
        df = df[df['site_no'] == site_no]
        print(f"Filtered to site_no={site_no}: {len(df)} samples")

    # Handle date column
    if 'Time' in df.columns:
        df['date'] = pd.to_datetime(df['Time'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values('date').reset_index(drop=True)

    # Handle missing values - only interpolate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill().ffill()

    target = df[target_col].values.astype(np.float64)

    return df, target


class IMFDataset(Dataset):
    """
    Dataset for single IMF with change-aware features and event flags.

    Input: [IMF_value, Δx, |Δx|, rolling_std, rolling_zscore] (5 features)
    Event flags are provided separately for EventWeightedLoss.
    Abs_delta (|Δx|) is provided separately for AdaptiveWeightedLoss.
    """

    def __init__(
        self,
        imf: np.ndarray,
        features: np.ndarray,
        event_flags: np.ndarray,
        seq_len: int,
        pred_len: int,
        flag: str = 'train',
        scaler: Optional[StandardScaler] = None,
        feature_scaler: Optional[StandardScaler] = None
    ):
        """
        Args:
            imf: 1D array of IMF values
            features: 2D array of change-aware features (from original signal)
            event_flags: 1D array of event flags
            seq_len: Input sequence length
            pred_len: Prediction horizon
            flag: 'train', 'val', or 'test'
            scaler: Pre-fitted scaler for IMF
            feature_scaler: Pre-fitted scaler for features
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag

        # Calculate split boundaries (60/20/20)
        n = len(imf)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)

        border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
        border2s = [n_train, n_train + n_val, n]

        type_map = {'train': 0, 'val': 1, 'test': 2}
        idx = type_map[flag]

        # Scale IMF values using RobustScaler (preserves outliers better)
        if scaler is None:
            self.scaler = RobustScaler()  # Robust to outliers
            train_imf = imf[:n_train].reshape(-1, 1)
            self.scaler.fit(train_imf)
        else:
            self.scaler = scaler

        imf_scaled = self.scaler.transform(imf.reshape(-1, 1))

        # Scale features using RobustScaler
        if feature_scaler is None:
            self.feature_scaler = RobustScaler()  # Robust to outliers
            train_features = features[:n_train]
            self.feature_scaler.fit(train_features)
        else:
            self.feature_scaler = feature_scaler

        features_scaled = self.feature_scaler.transform(features)

        # Combine IMF with features: [IMF, Δx, |Δx|, rolling_std, rolling_zscore]
        combined = np.concatenate([imf_scaled, features_scaled], axis=1)

        # Get split
        self.data_x = combined[border1s[idx]:border2s[idx]]
        self.data_y = imf_scaled[border1s[idx]:border2s[idx]]  # Target is just IMF
        self.event_flags = event_flags[border1s[idx]:border2s[idx]]

        # Store |Δx| for AdaptiveWeightedLoss (column 1 of features = Abs_Delta_X)
        # Note: This is from the ORIGINAL signal, not scaled
        self.abs_delta = features[border1s[idx]:border2s[idx], 1]  # |Δx| column

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_event = self.event_flags[r_begin:r_end]
        seq_abs_delta = self.abs_delta[r_begin:r_end]  # |Δx| for prediction window

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_event, dtype=torch.float32),
            torch.tensor(seq_abs_delta, dtype=torch.float32)
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse scale predictions."""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def create_dataloaders(
    imf: np.ndarray,
    features: np.ndarray,
    event_flags: np.ndarray,
    seq_len: int,
    pred_len: int,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Create train, val, test dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, imf_scaler, feature_scaler)
    """
    # Create train dataset first to get scalers
    train_dataset = IMFDataset(
        imf, features, event_flags, seq_len, pred_len, flag='train'
    )
    imf_scaler = train_dataset.scaler
    feature_scaler = train_dataset.feature_scaler

    # Create val and test with same scalers
    val_dataset = IMFDataset(
        imf, features, event_flags, seq_len, pred_len,
        flag='val', scaler=imf_scaler, feature_scaler=feature_scaler
    )
    test_dataset = IMFDataset(
        imf, features, event_flags, seq_len, pred_len,
        flag='test', scaler=imf_scaler, feature_scaler=feature_scaler
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, imf_scaler, feature_scaler


if __name__ == "__main__":
    print("Testing data loader...")

    # Generate test data
    np.random.seed(42)
    n = 1000
    imf = np.sin(np.linspace(0, 20, n)) + 0.1 * np.random.randn(n)
    features = np.random.randn(n, 4)  # 4 change-aware features
    event_flags = (np.random.rand(n) > 0.95).astype(np.float32)

    train_loader, val_loader, test_loader, scaler, feat_scaler = create_dataloaders(
        imf, features, event_flags, seq_len=168, pred_len=24, batch_size=32
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check batch
    x, y, e, abs_d = next(iter(train_loader))
    print(f"Input shape: {x.shape}")       # (batch, seq_len, 5)
    print(f"Target shape: {y.shape}")      # (batch, pred_len, 1)
    print(f"Event shape: {e.shape}")       # (batch, pred_len)
    print(f"Abs_delta shape: {abs_d.shape}") # (batch, pred_len)
