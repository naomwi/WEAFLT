import torch

CONFIG = {
    'seq_len': 96,
    'pred_len': 24,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 7,
    'event_weight': 5.0,
    'targets': ['Turbidity_log', 'Flow_log', 'EC_log', 'DO', 'pH', 'Temp'],
    'csv_path': "datasets/New_data/Training_data/Final_Dataset_Multivariate_CEEMDAN.csv"
}
RUNTIME_LOG = []
STABILITY_LOG = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
