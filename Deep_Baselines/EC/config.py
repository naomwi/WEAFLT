"""
Configuration for Deep Baselines
End-to-end models: LSTM, PatchTST, Transformer (NO decomposition)
Target: EC
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
MODEL_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

# Add project root to path for importing config_global
sys.path.insert(0, str(PROJECT_DIR))
from config_global import (
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    DATA_CONFIG as GLOBAL_DATA_CONFIG,
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
    'target_col': 'EC',
    'train_ratio': GLOBAL_DATA_CONFIG['train_ratio'],
    'val_ratio': GLOBAL_DATA_CONFIG['val_ratio'],
    'test_ratio': GLOBAL_DATA_CONFIG['test_ratio'],
    'seq_len': GLOBAL_DATA_CONFIG['seq_len'],
    'batch_size': {
        'lstm': get_batch_size('lstm'),           # 256
        'transformer': get_batch_size('transformer'),  # 128
        'patchtst': get_batch_size('patchtst'),   # 128
    },
    'batch_size_eval': GLOBAL_DATA_CONFIG['batch_size_eval'],
}

# Re-export HORIZONS
HORIZONS = HORIZONS

# =============================================================================
# MODEL CONFIGURATION - RTX 3090 Optimized
# =============================================================================
MODEL_CONFIG = {
    'lstm': GLOBAL_MODEL_CONFIG['lstm'],           # hidden=512, layers=3
    'transformer': GLOBAL_MODEL_CONFIG['transformer'],  # d_model=512, layers=4
    'patchtst': GLOBAL_MODEL_CONFIG['patchtst'],   # d_model=512, layers=4
}

# =============================================================================
# TRAINING CONFIGURATION - RTX 3090 Optimized
# =============================================================================
TRAIN_CONFIG = GLOBAL_TRAIN_CONFIG.copy()

# =============================================================================
# EXPERIMENTS
# =============================================================================
EXPERIMENTS = {
    'lstm': {'description': 'LSTM (end-to-end)'},
    'patchtst': {'description': 'PatchTST (end-to-end)'},
    'transformer': {'description': 'Transformer (end-to-end)'},
}


def get_batch_size_for_model(model_type: str) -> int:
    """Get batch size for specific model type."""
    return DATA_CONFIG['batch_size'].get(model_type, 128)


def get_config_summary():
    print("\n" + "=" * 60)
    print("DEEP BASELINES (End-to-End) - RTX 3090 Optimized")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Target: {DATA_CONFIG['target_col']}")
    print(f"Models: LSTM, PatchTST, Transformer")
    print(f"Decomposition: None (direct from raw signal)")
    print(f"Loss: MSE")
    print(f"\nModel Sizes (RTX 3090):")
    print(f"  LSTM: hidden={MODEL_CONFIG['lstm']['hidden_size']}, layers={MODEL_CONFIG['lstm']['num_layers']}")
    print(f"  Transformer: d_model={MODEL_CONFIG['transformer']['d_model']}, layers={MODEL_CONFIG['transformer']['num_layers']}")
    print(f"  PatchTST: d_model={MODEL_CONFIG['patchtst']['d_model']}, layers={MODEL_CONFIG['patchtst']['num_layers']}")
    print(f"\nBatch Sizes:")
    for model, bs in DATA_CONFIG['batch_size'].items():
        print(f"  {model}: {bs}")
    print(f"\nTraining:")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  AMP: {TRAIN_CONFIG['use_amp']}")
    print("=" * 60)


if __name__ == "__main__":
    get_config_summary()
