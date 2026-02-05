import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Hàm đọc và gộp dữ liệu
def load_raw_data(ec_path, ph_path):
    df_ec = pd.read_csv(ec_path, parse_dates=['date'])
    df_ph = pd.read_csv(ph_path, parse_dates=['date'])
    
    # Rename
    df_ec = df_ec.rename(columns={'OT': 'EC'})
    df_ph = df_ph.rename(columns={'OT': 'pH'})
    
    # Merge
    df = pd.merge(df_ec, df_ph, on='date', how='inner').sort_values('date')
    
    # Fill missing values (using bfill() instead of deprecated fillna(method='bfill'))
    df = df.interpolate(method='linear').bfill()
    return df

class IMFDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, flag='train'):
        # data: mảng 1 chiều của 1 IMF cụ thể
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Chia 70% Train, 10% Val, 20% Test
        n_train = int(len(data) * 0.7)
        n_test = int(len(data) * 0.2)
        n_val = len(data) - n_train - n_test
        
        border1s = [0, n_train - seq_len, len(data) - n_test - seq_len]
        border2s = [n_train, n_train + n_val, len(data)]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        idx = type_map[flag]
        
        # Scale dữ liệu (StandardScaler)
        self.scaler = StandardScaler()
        train_data = data[border1s[0]:border2s[0]].reshape(-1, 1)
        self.scaler.fit(train_data)
        
        # Transform
        data_scaled = self.scaler.transform(data.reshape(-1, 1))
        
        self.data_x = data_scaled[border1s[idx]:border2s[idx]]
        self.data_y = data_scaled[border1s[idx]:border2s[idx]]
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse(self, data):
        return self.scaler.inverse_transform(data)