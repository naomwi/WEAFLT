import numpy as np
import pandas as pd

def compute_event_threshold_from_train(df_train, target_col, percentile=95):
    series = df_train[target_col]
    threshold = np.percentile(series, percentile)

    return threshold

def detect_events_from_threshold(df, target_col, threshold):
    df = df.copy()
    
    base_name = target_col.replace('_log', '')
    event_col_name = f"{base_name}_is_extreme"
    
    df[event_col_name] = (df[target_col] >= threshold).astype(float)
    
    n_events = df[event_col_name].sum()    
    return df