"""
CEEMD Decomposition Module
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import os


def run_ceemd(
    data: np.ndarray,
    trials: int = 50,
    noise_width: float = 0.2,
    max_imfs: int = 12,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """CEEMD decomposition using EEMD from PyEMD."""
    from PyEMD import EEMD

    if verbose:
        print(f"  Running CEEMD with {trials} trials...")

    eemd = EEMD(trials=trials, noise_width=noise_width)
    imfs = eemd.eemd(data.reshape(-1), max_imf=max_imfs)

    if verbose:
        print(f"  Got {len(imfs)} components.")

    if len(imfs) > max_imfs:
        imf_components = imfs[:max_imfs]
        residue = imfs[max_imfs:].sum(axis=0)
    else:
        imf_components = imfs[:-1] if len(imfs) > 1 else imfs
        residue = imfs[-1] if len(imfs) > 1 else np.zeros_like(data)

    while len(imf_components) < max_imfs:
        imf_components = np.vstack([imf_components, np.zeros_like(data)])

    return imf_components[:max_imfs], residue


def save_imfs(imfs, residue, cache_dir, prefix="ec"):
    os.makedirs(cache_dir, exist_ok=True)
    for i, imf in enumerate(imfs):
        np.save(cache_dir / f"{prefix}_imf_{i+1}.npy", imf)
    np.save(cache_dir / f"{prefix}_residue.npy", residue)


def load_cached_imfs(cache_dir, prefix="ec", n_imfs=12):
    imfs = [np.load(cache_dir / f"{prefix}_imf_{i+1}.npy") for i in range(n_imfs)]
    residue = np.load(cache_dir / f"{prefix}_residue.npy")
    return {'imfs': np.array(imfs), 'residue': residue}


def get_or_create_imfs(data, cache_dir, prefix="ec", n_imfs=12, force_recompute=False, **kwargs):
    if not force_recompute:
        try:
            return load_cached_imfs(cache_dir, prefix, n_imfs)
        except FileNotFoundError:
            pass
    imfs, residue = run_ceemd(data, max_imfs=n_imfs, **kwargs)
    save_imfs(imfs, residue, cache_dir, prefix)
    return {'imfs': imfs, 'residue': residue}
