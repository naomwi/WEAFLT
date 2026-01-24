import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from src.Utils.path import DATA_DIR,OUTPUT_DIR
from src.Data.data_preprocessing import preprocess_dataframe
import numpy as np

def load_data_source_separate():
    """Load from separate files"""
    print("\n Loading separate files...")
    try:
        df = pd.read_csv(DATA_DIR/'New_data/USGS/water_data_2021_2025_clean.csv')
        all_features = ['Temp', 'Flow', 'EC', 'DO', 'pH', 'Turbidity']
        cols_need_log = ['Flow', 'Turbidity', 'EC']
        process_list = []

        for site,site_df in df.groupby('site_no'): 
            print(f'Station {site} in progress',end='')    
            site_process = preprocess_dataframe(
                site_df,
                features = all_features,
                log_cols = cols_need_log)
            
            process_list.append(site_process)
            print('DONE')
        
        final_df = pd.concat(process_list,axis=0)
        final_df.to_csv(DATA_DIR/'New_data/Training_data/input.csv',index=False)
        return final_df

    except FileNotFoundError:
        print("Error: Missing Data File")
        return None, None

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
    
def generate_window_data(df,feature_cols,target_cols,event_cols,label_cols,seq_len,pred_len):
    """
    Hàm này chạy qua từng site_no và cắt cửa sổ riêng biệt tránh lai tạp dữ liệu
    """
    X_list,y_list,event_list,label_list = [],[],[],[]

    for site_id, site_df in df.groupby('site_no'):
        data_x = site_df[feature_cols].values
        data_y = site_df[target_cols].values

        if event_cols in site_df.columns:
            data_event = site_df[event_cols].values
        else:
            data_event = np.zeros((len(site_df),1))
    
        if label_cols in site_df.columns:
            data_label = site_df[label_cols].values
        else:
            data_label = np.zeros((len(site_df),1))
        
        num_samples = len(site_df) - seq_len - pred_len + 1

        if num_samples > 0:
            for i in range(num_samples):
                X_list.append(data_x[i:i+seq_len])
                y_list.append(data_y[i+seq_len: i+seq_len+pred_len])
                event_list.append(data_event[i + seq_len: i+seq_len+pred_len])
                label_list.append(data_label[i+seq_len:i+seq_len+pred_len])
        
    if len(X_list) > 0:
        return (
            torch.FloatTensor(np.array(X_list)),
            torch.FloatTensor(np.array(y_list)),
            torch.FloatTensor(np.array(event_list)),
            torch.FloatTensor(np.array(label_list)),
        )
    else: return None

def create_dataloaders_advanced(df, target_cols, seq_len, pred_len, batch_size=32):
    
    print(f"\n Loading New Datasets...")
    excluded_input_cols = [
        'date', 'datetime', 'site_no', 'residue', 'Residue', 
        'Final_Label', 'Unnamed: 0'
    ]
    
    
    feature_cols = [c for c in df.columns if c not in excluded_input_cols and np.issubdtype(df[c].dtype, np.number)]

    main_target = target_cols[0] # Ví dụ: 'Turbidity_log'

    if "_log" in main_target:
        base_name = main_target.replace("_log", "") 
        event_col = f"{base_name}_is_extreme"       
    else:
        event_col = f"{main_target}_is_extreme"
    
    
    total_rows = len(df)
    train_size = int(0.6*total_rows)
    val_size = int(0.2*total_rows)

    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:train_size + val_size].copy()
    df_test = df.iloc[val_size:].copy()

    # Chuẩn hóa dữ liệu trên tập Train

    x_mean = df_train[feature_cols].mean()
    x_std = df_train[feature_cols].std()

    y_mean = df_train[target_cols].mean()
    y_std = df_train[target_cols].std().replace(0,1)

    def norm_apply(dataframe):
        dataframe[feature_cols] = (dataframe[feature_cols] - x_mean)/x_std
        dataframe[target_cols] = (dataframe[target_cols] - y_mean)/y_std
        return dataframe
    
    df_train = norm_apply(df_train)
    df_val = norm_apply(df_val)
    df_test = norm_apply(df_test)

    # Cắt cửa sổ dữ liệu

    train_tensors = generate_window_data(df_train,feature_cols,target_cols,event_col, "Final_Label", seq_len, pred_len)
    val_tensors = generate_window_data(df_val,feature_cols,target_cols,event_col, "Final_Label", seq_len, pred_len)
    test_tensors = generate_window_data(df_test,feature_cols,target_cols,event_col, "Final_Label", seq_len, pred_len)

    split_info = {
        'n_features': len(feature_cols),
        'n_targets': len(target_cols),
        'scaler': {
            'x_mean': torch.FloatTensor(x_mean.values),
            'x_std': torch.FloatTensor(x_std.values),
            'y_mean': torch.FloatTensor(y_mean.values),
            'y_std': torch.FloatTensor(y_std.values)
        }
    }
    
    # Tạo loaders
    train_set = CEEMDAN_WaterDataset(*train_tensors)
    val_set = CEEMDAN_WaterDataset(*val_tensors)
    test_set = CEEMDAN_WaterDataset(*test_tensors)

    loaders = (
        DataLoader(train_set,batch_size=batch_size,shuffle=False),
        DataLoader(val_set,batch_size=batch_size,shuffle=False),
        DataLoader(test_set,batch_size=batch_size,shuffle=False)
    )

    return loaders,None,split_info

    