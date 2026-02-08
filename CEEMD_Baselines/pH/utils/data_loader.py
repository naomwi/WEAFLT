"""
Data Loader for CEEMD Baselines
Standard IMF dataset (1 feature only)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def load_raw_data(data_path: str, target_col: str, site_no: int = 1463500):
    """
    Load raw data from CSV file and filter by station.

    Args:
        data_path: Path to CSV file
        target_col: Target column name
        site_no: USGS site number (default: 1463500, same as CEEMDAN_models)
    """
    df = pd.read_csv(data_path)

    # Filter by site_no (same as CEEMDAN_models)
    if 'site_no' in df.columns and site_no is not None:
        df = df[df['site_no'] == site_no]
        print(f"Filtered to site_no={site_no}: {len(df)} samples")

    if 'Time' in df.columns:
        df['date'] = pd.to_datetime(df['Time'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Handle missing values - only interpolate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill().ffill()

    return df, df[target_col].values.astype(np.float64)


class IMFDataset(Dataset):
    """
    Standard IMF dataset - NO SCALING.

    Important: CEEMDAN/CEEMD decomposition preserves scale:
    original_signal = sum(IMFs) + Residue (exact mathematical property)

    Therefore, we do NOT scale IMFs. Predictions can be summed directly
    to reconstruct the original signal prediction.
    """

    def __init__(self, imf, seq_len, pred_len, flag='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len

        n = len(imf)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)

        border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
        border2s = [n_train, n_train + n_val, n]

        idx = {'train': 0, 'val': 1, 'test': 2}[flag]

        # NO SCALING - keep original IMF scale
        # This allows sum(IMF_preds) + Residue_pred = Original_signal_pred
        self.data = imf[border1s[idx]:border2s[idx]].reshape(-1, 1)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        return (
            torch.tensor(self.data[index:s_end], dtype=torch.float32),
            torch.tensor(self.data[s_end:r_end], dtype=torch.float32)
        )


def create_dataloaders(imf, seq_len, pred_len, batch_size=64):
    """
    Create train/val/test dataloaders for a single IMF.

    Note: No scaling is applied. Returns None for scaler (backward compatibility).
    """
    train_ds = IMFDataset(imf, seq_len, pred_len, 'train')
    val_ds = IMFDataset(imf, seq_len, pred_len, 'val')
    test_ds = IMFDataset(imf, seq_len, pred_len, 'test')

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        None  # No scaler - IMFs are not scaled
    )
