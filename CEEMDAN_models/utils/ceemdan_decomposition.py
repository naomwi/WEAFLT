"""
CEEMDAN Decomposition Module
Decomposes time series into IMFs (Intrinsic Mode Functions) + Residue
Simple implementation - saves all components to files
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PyEMD import CEEMDAN
import os

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CEEMDAN_CONFIG, IMF_DIR


def decompose_series(series: np.ndarray, n_imfs: int = 12, verbose: bool = True) -> dict:
    """
    Decompose a single time series using CEEMDAN.

    Args:
        series: 1D numpy array of the time series
        n_imfs: Maximum number of IMFs to extract
        verbose: Print progress

    Returns:
        dict with 'imfs' (list of IMF arrays) and 'residue' (1D array)
    """
    if verbose:
        print(f"  Starting CEEMDAN with {CEEMDAN_CONFIG['trials']} trials...")
        print(f"  This may take a while for {len(series)} data points...")

    ceemdan = CEEMDAN(
        trials=CEEMDAN_CONFIG['trials'],
        epsilon=CEEMDAN_CONFIG['noise_width'],
        parallel=True,  # Enable parallel processing
        processes=None,  # Use all available cores
    )

    # Perform decomposition
    imfs = ceemdan.ceemdan(series, max_imf=n_imfs)

    if verbose:
        print(f"  Decomposition complete! Got {len(imfs)} components.")

    # Separate IMFs and residue
    # Last component is typically the residue (trend)
    if len(imfs) > n_imfs:
        imf_components = imfs[:n_imfs]
        residue = imfs[n_imfs:].sum(axis=0)
    else:
        imf_components = imfs[:-1] if len(imfs) > 1 else imfs
        residue = imfs[-1] if len(imfs) > 1 else np.zeros_like(series)

    # Pad to exactly n_imfs if needed
    while len(imf_components) < n_imfs:
        imf_components = np.vstack([imf_components, np.zeros_like(series)])

    return {
        'imfs': imf_components[:n_imfs],  # Exactly n_imfs components
        'residue': residue
    }


def decompose_and_save(data: pd.Series, save_dir: Path, prefix: str = "ec"):
    """
    Decompose time series and save all IMFs + residue to files.

    Args:
        data: pandas Series with the target variable
        save_dir: Directory to save IMF files
        prefix: Prefix for filenames

    Returns:
        dict with paths to saved files
    """
    os.makedirs(save_dir, exist_ok=True)

    values = data.values.astype(np.float64)
    n_imfs = CEEMDAN_CONFIG['n_imfs']

    print(f"Decomposing {len(values)} data points into {n_imfs} IMFs + residue...")

    result = decompose_series(values, n_imfs=n_imfs, verbose=True)

    saved_files = {}

    # Save each IMF
    for i, imf in enumerate(result['imfs']):
        filename = save_dir / f"{prefix}_imf_{i+1}.npy"
        np.save(filename, imf)
        saved_files[f'imf_{i+1}'] = filename
        print(f"  Saved IMF {i+1}: {filename}")

    # Save residue
    residue_file = save_dir / f"{prefix}_residue.npy"
    np.save(residue_file, result['residue'])
    saved_files['residue'] = residue_file
    print(f"  Saved Residue: {residue_file}")

    # Verify reconstruction
    reconstructed = np.sum(result['imfs'], axis=0) + result['residue']
    reconstruction_error = np.mean(np.abs(values - reconstructed))
    print(f"  Reconstruction error: {reconstruction_error:.6f}")

    return saved_files, result


def load_imfs(load_dir: Path, prefix: str = "ec", n_imfs: int = 12) -> dict:
    """
    Load previously saved IMFs and residue.

    Args:
        load_dir: Directory containing IMF files
        prefix: Prefix used when saving
        n_imfs: Number of IMFs to load

    Returns:
        dict with 'imfs' array and 'residue' array
    """
    imfs = []
    for i in range(n_imfs):
        filename = load_dir / f"{prefix}_imf_{i+1}.npy"
        if filename.exists():
            imfs.append(np.load(filename))
        else:
            raise FileNotFoundError(f"IMF file not found: {filename}")

    residue_file = load_dir / f"{prefix}_residue.npy"
    if residue_file.exists():
        residue = np.load(residue_file)
    else:
        raise FileNotFoundError(f"Residue file not found: {residue_file}")

    return {
        'imfs': np.array(imfs),
        'residue': residue
    }


def reconstruct_from_predictions(imf_predictions: list, residue_prediction: np.ndarray) -> np.ndarray:
    """
    Reconstruct the final prediction by summing all IMF predictions + residue prediction.

    Args:
        imf_predictions: List of predictions for each IMF
        residue_prediction: Prediction for residue

    Returns:
        Final reconstructed prediction
    """
    total = np.zeros_like(residue_prediction)

    for imf_pred in imf_predictions:
        total += imf_pred

    total += residue_prediction

    return total


if __name__ == "__main__":
    # Test decomposition
    print("Testing CEEMDAN decomposition...")

    # Generate test signal
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))

    result = decompose_series(signal, n_imfs=5)
    print(f"Number of IMFs: {len(result['imfs'])}")
    print(f"Residue shape: {result['residue'].shape}")

    # Verify reconstruction
    reconstructed = np.sum(result['imfs'], axis=0) + result['residue']
    error = np.mean(np.abs(signal - reconstructed))
    print(f"Reconstruction error: {error:.6f}")
