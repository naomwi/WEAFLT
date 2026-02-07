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


def load_raw_data(data_path: str, target_col: str):
    df = pd.read_csv(data_path)
    if 'Time' in df.columns:
        df['date'] = pd.to_datetime(df['Time'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.interpolate(method='linear').bfill().ffill()
    return df, df[target_col].values.astype(np.float64)


class IMFDataset(Dataset):
    """Standard IMF dataset - 1 feature only."""

    def __init__(self, imf, seq_len, pred_len, flag='train', scaler=None):
        self.seq_len = seq_len
        self.pred_len = pred_len

        n = len(imf)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)

        border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
        border2s = [n_train, n_train + n_val, n]

        idx = {'train': 0, 'val': 1, 'test': 2}[flag]

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(imf[:n_train].reshape(-1, 1))
        else:
            self.scaler = scaler

        imf_scaled = self.scaler.transform(imf.reshape(-1, 1))
        self.data = imf_scaled[border1s[idx]:border2s[idx]]

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


def create_dataloaders(imf, seq_len, pred_len, batch_size=64):
    train_ds = IMFDataset(imf, seq_len, pred_len, 'train')
    val_ds = IMFDataset(imf, seq_len, pred_len, 'val', train_ds.scaler)
    test_ds = IMFDataset(imf, seq_len, pred_len, 'test', train_ds.scaler)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        train_ds.scaler
    )
