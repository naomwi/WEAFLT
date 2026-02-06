"""
Data Loading Module for CEEMDAN-based Forecasting
Simple data loading and preprocessing - NO advanced features
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_CONFIG, IMF_DIR, DATA_DIR, CEEMDAN_CONFIG


class TimeSeriesDataset(Dataset):
    """Simple time series dataset for sequence-to-sequence prediction."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        """
        Args:
            data: 1D numpy array of values
            seq_len: Input sequence length
            pred_len: Prediction horizon length
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x.unsqueeze(-1), y.unsqueeze(-1)  # Add feature dimension


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        data_path: Path to CSV file. If None, searches in default locations.

    Returns:
        pandas DataFrame with the data
    """
    if data_path is not None and Path(data_path).exists():
        return pd.read_csv(data_path)

    # Search for data in common locations
    possible_paths = [
        DATA_DIR / "water_data_2021_2025_clean.csv",
        DATA_DIR / "USGS" / "water_data_2021_2025_clean.csv",
        Path(__file__).resolve().parents[2] / "Baselines_model" / "data" / "USGs" / "water_data_2021_2025_clean.csv",
        Path(__file__).resolve().parents[2] / "Full_model" / "data" / "USGs" / "water_data_2021_2025_clean.csv",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Loading data from: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(f"Data file not found. Tried: {possible_paths}")


def prepare_data_splits(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets chronologically.

    Args:
        data: 1D numpy array
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        test_ratio: Ratio for test data

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data


class IMFDataManager:
    """Manages loading and processing of IMF data for training."""

    def __init__(self, n_imfs: int = 12):
        self.n_imfs = n_imfs
        self.scalers = {}  # One scaler per component

    def load_and_scale_component(
        self,
        component_data: np.ndarray,
        component_name: str,
        fit_scaler: bool = True
    ) -> np.ndarray:
        """
        Scale a single component (IMF or residue).

        Args:
            component_data: 1D array of component values
            component_name: Name for scaler lookup (e.g., 'imf_1', 'residue')
            fit_scaler: Whether to fit a new scaler (True for train, False for val/test)

        Returns:
            Scaled data
        """
        data_2d = component_data.reshape(-1, 1)

        if fit_scaler:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data_2d).flatten()
            self.scalers[component_name] = scaler
        else:
            if component_name not in self.scalers:
                raise ValueError(f"Scaler for {component_name} not found. Fit on training data first.")
            scaled = self.scalers[component_name].transform(data_2d).flatten()

        return scaled

    def inverse_scale(self, scaled_data: np.ndarray, component_name: str) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        if component_name not in self.scalers:
            raise ValueError(f"Scaler for {component_name} not found.")

        data_2d = scaled_data.reshape(-1, 1)
        return self.scalers[component_name].inverse_transform(data_2d).flatten()

    def create_dataloaders(
        self,
        component_data: np.ndarray,
        component_name: str,
        seq_len: int,
        pred_len: int,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders for a single component.

        Args:
            component_data: Full time series for this component
            component_name: Name of the component
            seq_len: Input sequence length
            pred_len: Prediction horizon
            batch_size: Batch size

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split data
        train_data, val_data, test_data = prepare_data_splits(
            component_data,
            DATA_CONFIG['train_ratio'],
            DATA_CONFIG['val_ratio'],
            DATA_CONFIG['test_ratio']
        )

        # Scale data (fit on train only)
        train_scaled = self.load_and_scale_component(train_data, component_name, fit_scaler=True)
        val_scaled = self.load_and_scale_component(val_data, component_name, fit_scaler=False)
        test_scaled = self.load_and_scale_component(test_data, component_name, fit_scaler=False)

        # Create datasets
        train_dataset = TimeSeriesDataset(train_scaled, seq_len, pred_len)
        val_dataset = TimeSeriesDataset(val_scaled, seq_len, pred_len)
        test_dataset = TimeSeriesDataset(test_scaled, seq_len, pred_len)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


def get_target_data(df: pd.DataFrame, target_col: str = 'EC') -> np.ndarray:
    """
    Extract target column from dataframe.

    Args:
        df: DataFrame with data
        target_col: Name of target column

    Returns:
        1D numpy array of target values
    """
    # Handle different possible column names
    possible_names = [target_col, target_col.lower(), target_col.upper()]

    for name in possible_names:
        if name in df.columns:
            data = df[name].values
            # Remove NaN values
            data = data[~np.isnan(data)]
            print(f"Loaded {len(data)} values for target '{name}'")
            return data.astype(np.float64)

    raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    df = load_raw_data()
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")

    target = get_target_data(df, 'EC')
    print(f"Target shape: {target.shape}")
    print(f"Target stats: min={target.min():.2f}, max={target.max():.2f}, mean={target.mean():.2f}")
