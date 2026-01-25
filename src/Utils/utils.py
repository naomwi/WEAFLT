import numpy as np
import pandas as pd



def compute_threshold_and_detect_events(df, target_cols, split_percent, percentile=95):
    data = df.copy()
    
    train_mask = int(len(data)*split_percent)
    df_train = data[:train_mask]
    
    if len(df_train) == 0:
        print("Warning: Train set rỗng, kiểm tra lại split_date!")
        return data

    feature_stats = {}

    for col in target_cols:
        abs_delta_col = f'{col}_abs_delta'
        
        if abs_delta_col not in df_train.columns:
            continue
            
        threshold_val = df_train[abs_delta_col].quantile(percentile / 100.0)
        feature_stats[col] = threshold_val
        
        data[f'{col}_event_flag'] = (data[abs_delta_col] > threshold_val).astype(int)
        
        print(f" - {col}: Threshold (Top {100-percentile}%) = {threshold_val:.4f}")
        
    return data