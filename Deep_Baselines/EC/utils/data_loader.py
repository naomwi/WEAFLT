"""
Data Loader for Deep Baselines
End-to-end from raw signal (no decomposition)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple


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
    # Handle missing values - compatible with older pandas versions
    try:
        df = df.infer_objects(copy=False).interpolate(method='linear').bfill().ffill()
    except TypeError:
        df = df.infer_objects().interpolate(method='linear').bfill().ffill()
    return df, df[target_col].values.astype(np.float64)


class TimeSeriesDataset(Dataset):
    """Dataset for end-to-end forecasting from raw signal."""

    def __init__(self, data, seq_len, pred_len, flag='train', scaler=None):
        self.seq_len = seq_len
        self.pred_len = pred_len

        n = len(data)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)

        border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
        border2s = [n_train, n_train + n_val, n]

        idx = {'train': 0, 'val': 1, 'test': 2}[flag]

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(data[:n_train].reshape(-1, 1))
        else:
            self.scaler = scaler

        data_scaled = self.scaler.transform(data.reshape(-1, 1))
        self.data = data_scaled[border1s[idx]:border2s[idx]]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        return (
            torch.tensor(self.data[index:s_end], dtype=torch.float32),
            torch.tensor(self.data[s_end:r_end], dtype=torch.float32)
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def create_dataloaders(data, seq_len, pred_len, batch_size=64):
    train_ds = TimeSeriesDataset(data, seq_len, pred_len, 'train')
    val_ds = TimeSeriesDataset(data, seq_len, pred_len, 'val', train_ds.scaler)
    test_ds = TimeSeriesDataset(data, seq_len, pred_len, 'test', train_ds.scaler)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        train_ds.scaler
    )
