import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch 
import torch.nn as nn
import pandas as pd
from PyEMD import CEEMDAN
import numpy as np
from tqdm import tqdm
from src.Utils.path import DATA_DIR
    
def apply_ceemd_decomposition(df, target_col, n_imfs=4,trials=10):
    """CEEMD decomposition"""
    print(f"\nApplying CEEMD...")
    
    data = df.copy()

    print(f"Starting Decompose on {len(target_col)} columns: {target_col}")
    imf_info ={}
    process_list = []
    for site_id,site_df in data.groupby('site_no'):
        print(f"Decompose {site_id}")
        for col in tqdm(target_col,desc='Processing Columns'):

            signal = site_df[col].values
            if np.isnan(signal).any():
                # Fill tạm để chạy thuật toán
                signal = pd.Series(signal).interpolate().bfill().ffill().values
            try:
                ceemdan = CEEMDAN(trials=trials,noise_scale=0.2,parallel=True)

                if len(signal) > 50:
                    ceemdan.ceemdan(signal,max_imf=n_imfs)
                    imfs,residue = ceemdan.get_imfs_and_residue()

                    for i in range(len(imfs)):
                        site_df[f'{col}_IMF_{i+1}'] = imfs[i]
                    site_df[f"{col}_residue"] = residue

                    imf_info[f"{site_id}_{col}"] = len(imfs)
                else:
                    print(f"signal of {col} too short!!")
                    imf_info[col] = 0
            except Exception as e:
                print(f"Error in decompose {col}: {str(e)}")
                residue = pd.Series(signal).rolling(window=50, center=True).mean().bfill().ffill().values
                noise = signal - residue
                
                site_df[f"{col}_IMF_1"] = noise
                site_df[f"{col}_Residue"] = residue
                imf_info[f"{site_id}_{col}"] = 1
            process_list.append(site_df)

    print("\n FINISH Decomposition!!")
    return pd.concat(process_list,ignore_index=True), imf_info