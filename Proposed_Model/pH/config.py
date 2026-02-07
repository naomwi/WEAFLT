"""
Configuration for Proposed Model: CA-CEEMDAN-LTSF
Target: pH
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
CEEMDAN_CACHE_DIR = ROOT_DIR / "cache"  # pH has its own cache
CACHE_DIR = ROOT_DIR / "cache"
MODEL_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

# Add project root to path for importing config_global
sys.path.insert(0, str(PROJECT_DIR))
from config_global import (
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    DATA_CONFIG as GLOBAL_DATA_CONFIG,
    DECOMPOSITION_CONFIG as GLOBAL_DECOMPOSITION_CONFIG,
    FEATURE_CONFIG as GLOBAL_FEATURE_CONFIG,
    MODEL_CONFIG as GLOBAL_MODEL_CONFIG,
    TRAIN_CONFIG as GLOBAL_TRAIN_CONFIG,
    LOSS_CONFIG as GLOBAL_LOSS_CONFIG,
    ACTIVE_LOSS_TYPE,
    HORIZONS,
    get_batch_size,
)

# =============================================================================
# DATA CONFIGURATION - RTX 3090 Optimized
# =============================================================================
DATA_CONFIG = {
    'data_file': GLOBAL_DATA_CONFIG['data_file'],
    'target_col': 'pH',                    # Target column - pH
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
# DECOMPOSITION CONFIGURATION
# =============================================================================
DECOMPOSITION_CONFIG = GLOBAL_DECOMPOSITION_CONFIG['ceemdan']
DECOMPOSITION_CONFIG['method'] = 'ceemdan'

# =============================================================================
# CHANGE-AWARE FEATURES CONFIGURATION
# =============================================================================
FEATURE_CONFIG = GLOBAL_FEATURE_CONFIG.copy()

# Change-aware feature names (computed from ORIGINAL signal, not IMF)
FEATURE_NAMES = ['IMF', 'Delta_X', 'Abs_Delta_X', 'Rolling_Std', 'Rolling_Zscore']
INPUT_DIM = len(FEATURE_NAMES)  # 5 features

# =============================================================================
# MODEL CONFIGURATION - RTX 3090 Optimized
# =============================================================================
MODEL_CONFIG = {
    'dlinear': GLOBAL_MODEL_CONFIG['dlinear'],
    'nlinear': GLOBAL_MODEL_CONFIG['nlinear'],
}

# =============================================================================
# LOSS CONFIGURATION
# =============================================================================
# Use active loss type from global config ('event_weighted' or 'adaptive')
LOSS_TYPE = ACTIVE_LOSS_TYPE

if LOSS_TYPE == 'adaptive':
    LOSS_CONFIG = GLOBAL_LOSS_CONFIG['adaptive'].copy()
    LOSS_CONFIG['type'] = 'adaptive'
else:
    LOSS_CONFIG = GLOBAL_LOSS_CONFIG['event_weighted'].copy()
    LOSS_CONFIG['type'] = 'event_weighted'

# =============================================================================
# TRAINING CONFIGURATION - RTX 3090 Optimized
# =============================================================================
TRAIN_CONFIG = GLOBAL_TRAIN_CONFIG.copy()

# =============================================================================
# EXPERIMENTS
# =============================================================================
EXPERIMENTS = {
    'dlinear': {
        'model': 'dlinear',
        'description': 'CA-CEEMDAN-DLinear with EV_Loss',
    },
    'nlinear': {
        'model': 'nlinear',
        'description': 'CA-CEEMDAN-NLinear with EV_Loss',
    },
}


def get_config_summary():
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("PROPOSED MODEL: CA-CEEMDAN-LTSF (RTX 3090 Optimized)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Target: {DATA_CONFIG['target_col']}")
    print(f"Split: {DATA_CONFIG['train_ratio']}/{DATA_CONFIG['val_ratio']}/{DATA_CONFIG['test_ratio']}")
    print(f"Seq Length: {DATA_CONFIG['seq_len']}")
    print(f"Batch Size: {DATA_CONFIG['batch_size']}")
    print(f"Horizons: {HORIZONS}")
    print(f"Decomposition: CEEMDAN")
    print(f"Input Features: {FEATURE_NAMES} ({INPUT_DIM} features)")

    # Print loss config based on type
    if LOSS_CONFIG['type'] == 'adaptive':
        print(f"Loss: {LOSS_CONFIG['type']} (alpha={LOSS_CONFIG['alpha']}, max_weight={LOSS_CONFIG['max_weight']})")
    else:
        print(f"Loss: {LOSS_CONFIG['type']} (weight={LOSS_CONFIG['event_weight']})")

    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"AMP: {TRAIN_CONFIG['use_amp']}")
    print("=" * 60)


if __name__ == "__main__":
    get_config_summary()
