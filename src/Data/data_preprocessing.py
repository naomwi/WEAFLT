import pandas as pd
import numpy as np



def preprocess_dataframe(df, features,log_cols = ['Flow', 'Turbidity', 'EC'],sampling_rate='H'):
    data = df.copy()
    
    if sampling_rate == 'H':
        windows = {
            '24h':24,
            '7d':168
        }
    
    for col in features:
        data[col] = pd.to_numeric(data[col],errors='coerce')
        if col not in log_cols:
            target_series = data[col]
            suffix = ""
        else:
            data[col] = data[col].clip(lower=0.01)
            target_series = np.log1p(data[col])
            suffix = "_log"
        
        base_col_name = f"{col}{suffix}"
        data[base_col_name] = target_series

        for w,size in windows.items():
            data[f"{col}_mean_{w}"] = target_series.rolling(window = size).mean().bfill()
            data[f"{col}_std_{w}"] = target_series.rolling(window = size).std().bfill()

        threshold_val = target_series.quantile(0.95)
        data[f"{col}_is_extreme"] = (target_series > threshold_val).astype(float)
    
    return data
