"""
GLOBAL CONFIGURATION - Tất cả experiments dùng chung
Optimized for NVIDIA RTX 3090 (24GB VRAM)
"""

import torch
from pathlib import Path

# =============================================================================
# DEVICE CONFIGURATION - RTX 3090 Optimized
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8                     # 3090 can handle more workers
PIN_MEMORY = True
CUDA_BENCHMARK = True               # cudnn.benchmark for consistent input sizes

# Enable TF32 for faster training on Ampere GPUs (3090)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# =============================================================================
# DATA CONFIGURATION - RTX 3090 (24GB VRAM)
# =============================================================================
DATA_CONFIG = {
    'data_file': 'water_data_2021_2025_clean.csv',
    'targets': ['EC', 'pH'],

    # Split ratios
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,

    # Sequence settings
    'seq_len': 168,                 # 7 days input

    # RTX 3090 optimized batch sizes (24GB VRAM)
    'batch_size': {
        'dlinear': 512,             # Linear models are lightweight
        'nlinear': 512,
        'lstm': 256,                # LSTM needs more memory
        'transformer': 128,         # Attention is memory heavy
        'patchtst': 128,
    },
    'batch_size_eval': 512,         # Larger for inference
}

# Prediction horizons
HORIZONS = [6, 12, 24, 48, 96, 168]

# =============================================================================
# DECOMPOSITION CONFIGURATION
# =============================================================================
DECOMPOSITION_CONFIG = {
    'ceemdan': {
        'trials': 20,
        'epsilon': 0.2,
        'max_imfs': 12,
    },
    'ceemd': {
        'trials': 50,
        'noise_width': 0.2,
        'max_imfs': 12,
    },
}

# =============================================================================
# CHANGE-AWARE FEATURES (Proposed Model)
# =============================================================================
FEATURE_CONFIG = {
    'rolling_std_window': 12,       # 12 hours
    'rolling_zscore_window': 24,    # 24 hours
    'event_percentile': 95.0,       # Top 5% = sudden fluctuations
}

# =============================================================================
# MODEL CONFIGURATIONS - RTX 3090 Optimized
# =============================================================================
MODEL_CONFIG = {
    # Linear models (can be larger on 3090)
    'dlinear': {
        'kernel_size': 7,  # Reduced from 25 to preserve outlier signals
    },
    'nlinear': {},

    # LSTM - RTX 3090 optimized
    'lstm': {
        'hidden_size': 512,         # Large (3090 can handle)
        'num_layers': 3,            # Deeper
        'dropout': 0.2,
        'bidirectional': False,
    },

    # Transformer - RTX 3090 optimized
    'transformer': {
        'd_model': 512,             # Large
        'nhead': 8,
        'num_layers': 4,            # Deeper
        'd_ff': 2048,               # 4x d_model
        'dropout': 0.1,
    },

    # PatchTST - RTX 3090 optimized
    'patchtst': {
        'patch_len': 16,
        'stride': 8,
        'd_model': 512,             # Large
        'nhead': 8,
        'num_layers': 4,            # Deeper
        'd_ff': 2048,
        'dropout': 0.1,
    },
}

# =============================================================================
# TRAINING CONFIGURATION - RTX 3090 Optimized
# =============================================================================
TRAIN_CONFIG = {
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    # Early stopping
    'early_stopping_patience': 15,
    'min_delta': 1e-6,              # Minimum improvement

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler': 'cosine',          # CosineAnnealingLR
    'scheduler_config': {
        'T_max': 100,
        'eta_min': 1e-6,
    },

    # Gradient clipping (stability)
    'grad_clip': 1.0,

    # Mixed precision - CRITICAL for 3090 speed
    'use_amp': True,                # 2x speedup on 3090
    'amp_dtype': 'float16',         # or 'bfloat16' for newer PyTorch

    # Gradient accumulation (if batch doesn't fit)
    'accumulation_steps': 1,        # Increase if OOM
}

# =============================================================================
# LOSS CONFIGURATION
# =============================================================================
LOSS_CONFIG = {
    'mse': {},
    'event_weighted': {
        'event_weight': 4.0,  # Upweight outliers/events (increased to 4.0)
    },
}

# =============================================================================
# EXPERIMENTS SUMMARY
# =============================================================================
EXPERIMENTS = {
    'proposed_dlinear': {
        'folder': 'Proposed_Model',
        'decomposition': 'ceemdan',
        'model': 'dlinear',
        'features': 'change_aware',
        'loss': 'event_weighted',
    },
    'proposed_nlinear': {
        'folder': 'Proposed_Model',
        'decomposition': 'ceemdan',
        'model': 'nlinear',
        'features': 'change_aware',
        'loss': 'event_weighted',
    },
    'ceemd_dlinear': {
        'folder': 'CEEMD_Baselines',
        'decomposition': 'ceemd',
        'model': 'dlinear',
        'features': 'standard',
        'loss': 'mse',
    },
    'ceemd_nlinear': {
        'folder': 'CEEMD_Baselines',
        'decomposition': 'ceemd',
        'model': 'nlinear',
        'features': 'standard',
        'loss': 'mse',
    },
    'deep_lstm': {
        'folder': 'Deep_Baselines',
        'decomposition': None,
        'model': 'lstm',
        'features': 'standard',
        'loss': 'mse',
    },
    'deep_transformer': {
        'folder': 'Deep_Baselines',
        'decomposition': None,
        'model': 'transformer',
        'features': 'standard',
        'loss': 'mse',
    },
    'deep_patchtst': {
        'folder': 'Deep_Baselines',
        'decomposition': None,
        'model': 'patchtst',
        'features': 'standard',
        'loss': 'mse',
    },
}


def get_batch_size(model_type: str) -> int:
    """Get optimal batch size for model type."""
    return DATA_CONFIG['batch_size'].get(model_type, 128)


def print_config_summary():
    """Print configuration summary."""
    print("\n" + "=" * 70)
    print("GLOBAL CONFIGURATION - RTX 3090 OPTIMIZED")
    print("=" * 70)

    print(f"\n[DEVICE]")
    print(f"  Device: {DEVICE}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")

    print(f"\n[DATA]")
    print(f"  Targets: {DATA_CONFIG['targets']}")
    print(f"  Split: {DATA_CONFIG['train_ratio']}/{DATA_CONFIG['val_ratio']}/{DATA_CONFIG['test_ratio']}")
    print(f"  Seq Length: {DATA_CONFIG['seq_len']}")
    print(f"  Batch Sizes: {DATA_CONFIG['batch_size']}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print(f"  Pin Memory: {PIN_MEMORY}")

    print(f"\n[TRAINING]")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Early Stopping: {TRAIN_CONFIG['early_stopping_patience']} epochs")
    print(f"  Scheduler: {TRAIN_CONFIG['scheduler']}")
    print(f"  Mixed Precision (AMP): {TRAIN_CONFIG['use_amp']}")
    print(f"  Gradient Clipping: {TRAIN_CONFIG['grad_clip']}")

    print(f"\n[MODELS - RTX 3090 Optimized]")
    for name, cfg in MODEL_CONFIG.items():
        if cfg:
            key_params = []
            if 'hidden_size' in cfg:
                key_params.append(f"hidden={cfg['hidden_size']}")
            if 'd_model' in cfg:
                key_params.append(f"d_model={cfg['d_model']}")
            if 'num_layers' in cfg:
                key_params.append(f"layers={cfg['num_layers']}")
            print(f"  {name}: {', '.join(key_params) if key_params else 'default'}")

    print(f"\n[EXPERIMENTS]")
    total_runs = len(EXPERIMENTS) * len(HORIZONS) * 2
    print(f"  {len(EXPERIMENTS)} experiments x {len(HORIZONS)} horizons x 2 targets = {total_runs} runs")

    # Time estimate for 3090
    print(f"\n[ESTIMATED TIME - RTX 3090]")
    print(f"  Proposed_Model: ~8 min")
    print(f"  CEEMD_Baselines: ~5 min")
    print(f"  Deep_Baselines: ~20 min")
    print(f"  TOTAL: ~35-45 min")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_config_summary()
