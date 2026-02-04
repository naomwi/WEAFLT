import torch

CONFIG = {
    # Training parameters
    'seq_len': 96,
    'pred_len': 24,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 7,
    'event_weight': 5.0,
    'seed': 42,
    # CEEMD (Complete Ensemble EMD) parameters - base approach
    'n_imfs': 12,
    'ceemd_trials': 100,
    # Data parameters
    'window_size': 24,
    'event_percentile': 80,
    'site_col': 'site_no',
    'time_col': 'Time',
    'log_vars': ['Flow', 'Turbidity', 'EC'],
    'normal_vars': ['Temp', 'DO', 'pH'],
    'split_date': '2024-01-01',
    'targets': ['Turbidity_log', 'Flow_log', 'EC_log', 'DO', 'pH', 'Temp'],
    'event_col_suffix': '_event_flag',
}
RUNTIME_LOG = []
STABILITY_LOG = []

# Device configuration with GPU check
def get_device():
    """Get the best available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    return device

device = get_device()
