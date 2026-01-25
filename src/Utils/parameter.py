import torch

CONFIG = {
    # training parameters
    'seq_len': 96,
    'pred_len': 24,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 7,
    'event_weight': 5.0,
    'seed':42,
    # CEEMDAN parameters
    'n_imfs': 4,
    'ceemd_trials': 10,
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
