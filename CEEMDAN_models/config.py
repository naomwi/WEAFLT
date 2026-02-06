"""
Configuration file for CEEMDAN-based forecasting
Simple, clean implementation - NO advanced features
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMF_DIR = ROOT_DIR / "decomposed_imfs"
MODEL_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

# CEEMDAN Configuration
CEEMDAN_CONFIG = {
    'n_imfs': 12,           # Number of IMFs (will also have 1 residue = 13 total components)
    'noise_width': 0.05,    # Noise standard deviation
    'trials': 20,           # Reduced for 217k data points (was 100)
}

# Data Configuration
DATA_CONFIG = {
    'target_col': 'EC',     # Electrical Conductivity
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'seq_len': 168,         # Input sequence length (7 days * 24 hours)
    'batch_size': 64,
    'max_samples': 50000,   # Limit data size for CEEMDAN (None = use all)
}

# Prediction Horizons
HORIZONS = [6, 12, 24, 48, 96, 168]

# Model Configuration
MODEL_CONFIG = {
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.1,
    },
    'transformer': {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 2,
        'dropout': 0.1,
    },
    'patchtst': {
        'patch_len': 16,
        'stride': 8,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dropout': 0.1,
    },
}

# Training Configuration - SIMPLE, NO ADVANCED FEATURES
TRAIN_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'loss': 'mse',          # Standard MSE loss only
}

# Available Models
AVAILABLE_MODELS = ['lstm', 'patchtst', 'transformer']
