import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from src.Utils.path import DATA_DIR,OUTPUT_DIR
import numpy as np


class CEEMDAN_WaterDataset(Dataset):
    """
    Khu vực chứa dữ liệu
    """
    def __init__ (self,X,y,events,labels):
        self.X = X
        self.y_val = y
        self.events = events
        self.y_label = labels

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index],self.y_val[index],self.events[index],self.y_label[index]

def generate_window_data(df, feature_cols, target_cols, event_cols, label_cols, seq_len, pred_len):
    """
    Hàm cắt cửa sổ hỗ trợ Multi-Channel Events
    """
    X_list, y_list, event_list, label_list = [], [], [], []

    for site_id, site_df in df.groupby('site_no'):
        data_x = site_df[feature_cols].values
        data_y = site_df[target_cols].values 
        missing_cols = [c for c in event_cols if c not in site_df.columns]
        
        if len(missing_cols) == 0:
            data_event = site_df[event_cols].values # Shape: [N, 6]
        else:
            data_event = np.zeros_like(data_y)

        if label_cols in site_df.columns:
            data_label = site_df[label_cols].values
        else:
            data_label = np.zeros((len(site_df), 1))
        
        num_samples = len(site_df) - seq_len - pred_len + 1

        if num_samples > 0:
            for i in range(num_samples):
                X_list.append(data_x[i : i + seq_len])
                y_list.append(data_y[i + seq_len : i + seq_len + pred_len])
                event_list.append(data_event[i + seq_len : i + seq_len + pred_len])
                label_list.append(data_label[i + seq_len : i + seq_len + pred_len])
        
    if len(X_list) > 0:
        return (
            torch.FloatTensor(np.array(X_list)),
            torch.FloatTensor(np.array(y_list)),
            torch.FloatTensor(np.array(event_list)), 
            torch.FloatTensor(np.array(label_list)),
        )
    else: 
        return None

def create_dataloaders_advanced(df, target_cols, seq_len, pred_len, batch_size=32,split_date_val='2024-01-01', split_date_test='2025-01-01'):
    
    print(f"\n Loading New Datasets...")
    excluded_input_cols = [
        'date', 'datetime', 'site_no', 'residue', 'Residue', 
        'Final_Label', 'Unnamed: 0', 'Time' 
    ]
    
    all_numeric = [c for c in df.columns if c not in excluded_input_cols and np.issubdtype(df[c].dtype, np.number)]
    event_cols_found = [c for c in all_numeric if 'event' in c or 'flag' in c or 'extreme' in c]
    feature_cols = [c for c in all_numeric if c not in event_cols_found]
    event_cols = []

    print(f"Input Features: {len(feature_cols)} (Excluded {len(event_cols_found)} event columns)")
    for col in target_cols:
        if "_log" in col:
            base_name = col.replace("_log", "")
        else:
            base_name = col
        evt_name = f"{col}_event_flag" 
        if evt_name not in df.columns:
            evt_name = f"{base_name}_event_flag" 
        event_cols.append(evt_name)
        
    print(f"Feature Cols: {feature_cols}")
    print(f"Target Cols: {target_cols}")
    print(f"Event Mask Cols: {event_cols}")
    time_col_name = 'Time' 
    if time_col_name not in df.columns:
         print("Warning: No Time column found. Using Index split (Risk of site mixing!)")
         total_rows = len(df)
         train_end = int(0.6 * total_rows)
         val_end = int(0.8 * total_rows)
         
         df_train = df.iloc[:train_end].copy()
         df_val = df.iloc[train_end:val_end].copy()
         df_test = df.iloc[val_end:].copy()
    else:
        df[time_col_name] = pd.to_datetime(df[time_col_name])
        df_train = df[df[time_col_name] < split_date_val].copy()
        df_val = df[(df[time_col_name] >= split_date_val) & (df[time_col_name] < split_date_test)].copy()
        df_test = df[df[time_col_name] >= split_date_test].copy()

    print(f"Train samples: {len(df_train)} | Val samples: {len(df_val)} | Test samples: {len(df_test)}")

    x_mean = df_train[feature_cols].mean()
    x_std = df_train[feature_cols].std().replace(0, 1) 

    y_mean = df_train[target_cols].mean()
    y_std = df_train[target_cols].std().replace(0, 1)

    def norm_apply(dataframe):
        df_scaled = dataframe.copy()
        df_scaled[feature_cols] = (df_scaled[feature_cols] - x_mean) / x_std
        df_scaled[target_cols] = (df_scaled[target_cols] - y_mean) / y_std
        return df_scaled
    
    df_train = norm_apply(df_train)
    df_val = norm_apply(df_val)
    df_test = norm_apply(df_test)

   
    train_tensors = generate_window_data(df_train, feature_cols, target_cols, event_cols, "Final_Label", seq_len, pred_len)
    val_tensors = generate_window_data(df_val, feature_cols, target_cols, event_cols, "Final_Label", seq_len, pred_len)
    test_tensors = generate_window_data(df_test, feature_cols, target_cols, event_cols, "Final_Label", seq_len, pred_len)

    if train_tensors is None or val_tensors is None:
        raise ValueError("Train or Val set is empty! Check split_date or data length.")

    train_set = CEEMDAN_WaterDataset(*train_tensors)
    val_set = CEEMDAN_WaterDataset(*val_tensors)
    
    if test_tensors is not None:
        test_set = CEEMDAN_WaterDataset(*test_tensors)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    loaders = (
        DataLoader(train_set, batch_size=batch_size, shuffle=True), 
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        test_loader
    )
    
    split_info = {
        'n_features': len(feature_cols),
        'n_targets': len(target_cols),
        'scaler': {
            'x_mean': torch.FloatTensor(x_mean.values),
            'x_std': torch.FloatTensor(x_std.values),
            'y_mean': torch.FloatTensor(y_mean.values),
            'y_std': torch.FloatTensor(y_std.values)
        },
        'target_names': target_cols
    }

    return loaders, None, split_info

    