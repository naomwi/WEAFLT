"""
Configuration for CEEMD Baselines
Target: pH
CEEMD + DLinear/NLinear + Standard features + MSE Loss
Optimized for RTX 3090
"""

import sys
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "Baselines_model" / "data" / "USGs"
CACHE_DIR = ROOT_DIR / "cache"
MODEL_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

# Add project root to path for importing config_global
sys.path.insert(0, str(PROJECT_DIR))
from config_global import (
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    DATA_CONFIG as GLOBAL_DATA_CONFIG,
    DECOMPOSITION_CONFIG as GLOBAL_DECOMPOSITION_CONFIG,
    MODEL_CONFIG as GLOBAL_MODEL_CONFIG,
    TRAIN_CONFIG as GLOBAL_TRAIN_CONFIG,
    HORIZONS,
    get_batch_size,
)

# =============================================================================
# DATA CONFIGURATION - RTX 3090 Optimized
# =============================================================================
DATA_CONFIG = {
    'data_file': GLOBAL_DATA_CONFIG['data_file'],
    'target_col': 'pH',  # Target: pH
    'train_ratio': GLOBAL_DATA_CONFIG['train_ratio'],
    'val_ratio': GLOBAL_DATA_CONFIG['val_ratio'],
    'test_ratio': GLOBAL_DATA_CONFIG['test_ratio'],
    'seq_len': GLOBAL_DATA_CONFIG['seq_len'],
    'batch_size': get_batch_size('dlinear'),  # 512 for linear models
    'batch_size_eval': GLOBAL_DATA_CONFIG['batch_size_eval'],
}

# Re-export HORIZONS
HORIZONS = HORIZONS

# =============================================================================
# DECOMPOSITION CONFIGURATION - CEEMD
# =============================================================================
DECOMPOSITION_CONFIG = GLOBAL_DECOMPOSITION_CONFIG['ceemd'].copy()
DECOMPOSITION_CONFIG['method'] = 'ceemd'

# =============================================================================
# MODEL CONFIGURATION - RTX 3090 Optimized
# =============================================================================
MODEL_CONFIG = {
    'dlinear': GLOBAL_MODEL_CONFIG['dlinear'],
    'nlinear': GLOBAL_MODEL_CONFIG['nlinear'],
}

# =============================================================================
# TRAINING CONFIGURATION - RTX 3090 Optimized
# =============================================================================
TRAIN_CONFIG = GLOBAL_TRAIN_CONFIG.copy()

# =============================================================================
# EXPERIMENTS
# =============================================================================
EXPERIMENTS = {
    'dlinear': {'model': 'dlinear', 'description': 'CEEMD + DLinear + MSE'},
    'nlinear': {'model': 'nlinear', 'description': 'CEEMD + NLinear + MSE'},
}


def get_config_summary():
    print("\n" + "=" * 60)
    print("CEEMD BASELINES (RTX 3090 Optimized)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Target: {DATA_CONFIG['target_col']}")
    print(f"Batch Size: {DATA_CONFIG['batch_size']}")
    print(f"Decomposition: CEEMD")
    print(f"Features: Standard (IMF only)")
    print(f"Loss: MSE")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"AMP: {TRAIN_CONFIG['use_amp']}")
    print("=" * 60)


if __name__ == "__main__":
    get_config_summary()
