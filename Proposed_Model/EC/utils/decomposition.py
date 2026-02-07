"""
CEEMDAN Decomposition Module
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import os


def run_ceemdan(
    data: np.ndarray,
    trials: int = 20,
    epsilon: float = 0.2,
    max_imfs: int = 12,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CEEMDAN (Complete Ensemble EMD with Adaptive Noise) decomposition.

    Args:
        data: 1D numpy array of time series
        trials: Number of ensemble trials
        epsilon: Noise standard deviation
        max_imfs: Maximum number of IMFs to extract
        verbose: Print progress

    Returns:
        Tuple of (imfs, residue) where:
            imfs: 2D array of shape (n_imfs, length)
            residue: 1D array of residual component
    """
    from PyEMD import CEEMDAN

    if verbose:
        print(f"  Running CEEMDAN with {trials} trials...")
        print(f"  Data length: {len(data)}")

    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)
    imfs = ceemdan(data.reshape(-1), max_imf=max_imfs)

    if verbose:
        print(f"  Decomposition complete! Got {len(imfs)} components.")

    # Separate IMFs and residue
    if len(imfs) > max_imfs:
        imf_components = imfs[:max_imfs]
        residue = imfs[max_imfs:].sum(axis=0)
    else:
        imf_components = imfs[:-1] if len(imfs) > 1 else imfs
        residue = imfs[-1] if len(imfs) > 1 else np.zeros_like(data)

    # Pad to exactly max_imfs if needed
    while len(imf_components) < max_imfs:
        imf_components = np.vstack([imf_components, np.zeros_like(data)])

    return imf_components[:max_imfs], residue


def save_imfs(
    imfs: np.ndarray,
    residue: np.ndarray,
    cache_dir: Path,
    prefix: str = "ec"
) -> Dict[str, Path]:
    """Save IMFs and residue to .npy files."""
    os.makedirs(cache_dir, exist_ok=True)
    saved_files = {}

    for i, imf in enumerate(imfs):
        filename = cache_dir / f"{prefix}_imf_{i+1}.npy"
        np.save(filename, imf)
        saved_files[f'imf_{i+1}'] = filename

    residue_file = cache_dir / f"{prefix}_residue.npy"
    np.save(residue_file, residue)
    saved_files['residue'] = residue_file

    return saved_files


def load_cached_imfs(
    cache_dir: Path,
    prefix: str = "ec",
    n_imfs: int = 12
) -> Dict[str, np.ndarray]:
    """Load previously saved IMFs and residue."""
    imfs = []
    for i in range(n_imfs):
        filename = cache_dir / f"{prefix}_imf_{i+1}.npy"
        if filename.exists():
            imfs.append(np.load(filename))
        else:
            raise FileNotFoundError(f"IMF file not found: {filename}")

    residue_file = cache_dir / f"{prefix}_residue.npy"
    if residue_file.exists():
        residue = np.load(residue_file)
    else:
        raise FileNotFoundError(f"Residue file not found: {residue_file}")

    return {
        'imfs': np.array(imfs),
        'residue': residue
    }


def get_or_create_imfs(
    data: np.ndarray,
    cache_dir: Path,
    prefix: str = "ec",
    n_imfs: int = 12,
    force_recompute: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Load cached IMFs if available, otherwise compute and cache."""
    if not force_recompute:
        try:
            print(f"Loading cached IMFs from {cache_dir}...")
            result = load_cached_imfs(cache_dir, prefix, n_imfs)
            print(f"  Loaded {len(result['imfs'])} IMFs from cache.")
            return result
        except FileNotFoundError:
            print(f"  Cache not found, computing decomposition...")

    imfs, residue = run_ceemdan(data, max_imfs=n_imfs, **kwargs)
    save_imfs(imfs, residue, cache_dir, prefix)

    return {'imfs': imfs, 'residue': residue}
