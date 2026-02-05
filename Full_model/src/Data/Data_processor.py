import pandas as pd
import numpy as np
from PyEMD import CEEMDAN  # Using CEEMDAN (Complete Ensemble EMD with Adaptive Noise)
from tqdm.auto import tqdm

class DataProcessor:
    def __init__(self, config):
        """
        config: Dictionary chứa cấu hình 
        """
        self.cfg = config
        self.target_cols = [] 
        self.imf_cols = []   
        
    def load_and_clean(self, df):
        print("Cleaning & Log Transform...")
        
        # Sắp xếp thời gian
        df[self.cfg['time_col']] = pd.to_datetime(df[self.cfg['time_col']])
        df = df.sort_values(by=[self.cfg['site_col'], self.cfg['time_col']]).reset_index(drop=True)
        
        # Xử lý Log Transform & Clean
        processed_vars = []
        for col in self.cfg['log_vars']:
            # Clip để tránh log(0) hoặc số âm
            df[col] = df[col].clip(lower=0.01)
            new_col = f"{col}_log"
            df[new_col] = np.log1p(df[col])
            processed_vars.append(new_col)
            
        # Thêm các biến không log (như Temp, pH)
        for col in self.cfg['normal_vars']:
            if col in df.columns:
                processed_vars.append(col)
            
        self.target_cols = processed_vars
        print(f"    Target columns: {self.target_cols}")
        return df

    def apply_ceemdan(self, df):
        """
        Apply CEEMDAN (Complete Ensemble EMD with Adaptive Noise) decomposition to target columns.
        This is the advanced approach - CEEMDAN-LTSF.
        """
        print(f"Applying CEEMDAN Decomposition (n_imfs={self.cfg['n_imfs']})...")

        process_list = []
        n_imfs = self.cfg['n_imfs']
        trials = self.cfg.get('ceemd_trials', 100)

        for site_id, site_df in df.groupby(self.cfg['site_col']):
            site_df = site_df.copy()

            for col in tqdm(self.target_cols, desc=f"Site {site_id}", leave=False):
                signal = site_df[col].values

                # Handle NaN values
                if np.isnan(signal).any():
                    signal = pd.Series(signal).interpolate().bfill().ffill().values

                for i in range(n_imfs):
                    col_name = f'{col}_IMF_{i+1}'
                    site_df[col_name] = 0.0
                    if col_name not in self.imf_cols: self.imf_cols.append(col_name)

                res_name = f'{col}_residue'
                site_df[res_name] = 0.0
                if res_name not in self.imf_cols: self.imf_cols.append(res_name)

                try:
                    if len(signal) < 50: raise ValueError("Signal too short")

                    # Use CEEMDAN (Complete Ensemble EMD with Adaptive Noise)
                    ceemdan = CEEMDAN(trials=trials, epsilon=0.2)
                    imfs = ceemdan(signal, max_imf=n_imfs)

                    # Separate IMFs and residue (last component is residue)
                    if len(imfs) > 1:
                        found_imfs = min(len(imfs) - 1, n_imfs)
                        for i in range(found_imfs):
                            site_df[f'{col}_IMF_{i+1}'] = imfs[i]
                        site_df[res_name] = imfs[-1]  # Last IMF as residue
                    else:
                        site_df[res_name] = imfs[0] if len(imfs) > 0 else signal

                except Exception as e:
                    # Fallback: simple rolling decomposition
                    temp_res = pd.Series(signal).rolling(window=24, center=True, min_periods=1).mean().bfill().ffill().values
                    temp_noise = signal - temp_res

                    site_df[f'{col}_IMF_1'] = temp_noise
                    site_df[res_name] = temp_res

            process_list.append(site_df)

        return pd.concat(process_list, ignore_index=True)

    # Alias for backward compatibility
    def apply_ceemd(self, df):
        """Backward compatible alias - now uses CEEMDAN"""
        return self.apply_ceemdan(df)

    def generate_stats(self, df):
        print("Generating Statistical Features...")
        
        # Danh sách cần tính stats: Cột gốc + Cột Residual
        # (Không cần tính cho IMF vì đã quyết định bỏ qua event cho IMF)
        cols_to_process = self.target_cols + [f"{c}_residue" for c in self.target_cols]
        
        # Gom nhóm theo site
        grouper = df.groupby(self.cfg['site_col'])
        
        for col in cols_to_process:
            if col not in df.columns: continue
                
            # Delta
            df[f'{col}_delta'] = grouper[col].diff().fillna(0)
            df[f'{col}_abs_delta'] = df[f'{col}_delta'].abs()
            
            # Rolling Stats
            win = self.cfg['window_size']
            roll_std = grouper[col].transform(lambda x: x.rolling(win).std()).fillna(0)
            roll_mean = grouper[col].transform(lambda x: x.rolling(win).mean()).fillna(0)
            
            df[f'{col}_roll_std'] = roll_std
            # Z-score
            df[f'{col}_roll_zscore'] = (df[col] - roll_mean) / (roll_std + 1e-6)
            
        return df

    def detect_events(self, df):
        print("Detecting Events (Train-based Threshold)...")
        
        train_mask = df[self.cfg['time_col']] < self.cfg['split_date']
        df_train = df.loc[train_mask]
        
        if len(df_train) == 0: 
            print("WARNING: Train set empty based on split_date. Calculating threshold on full data instead.")
            df_train = df

        for col in self.target_cols + [f"{c}_residue" for c in self.target_cols]:
            abs_delta_col = f'{col}_abs_delta'
            if abs_delta_col not in df.columns: continue
            threshold = df_train[abs_delta_col].quantile(self.cfg['event_percentile'] / 100.0)
            df[f'{col}_event_flag'] = (df[abs_delta_col] > threshold).astype(float)
            
        return df

    def run_pipeline(self, df_raw, type=1):
        """
        Run the data processing pipeline.

        Args:
            df_raw: Raw DataFrame
            type: Pipeline type
                1 - Full pipeline (CEEMDAN + Stats + Events)
                2 - CEEMDAN only
                3 - Stats only
                4 - Events only
        """
        df = self.load_and_clean(df_raw)
        if type == 1:
            df = self.apply_ceemdan(df)
            df = self.generate_stats(df)
            df = self.detect_events(df)
        elif type == 2:
            df = self.apply_ceemdan(df)
        elif type == 3:
            df = self.generate_stats(df)
        else:
            df = self.detect_events(df)
        print(">>> Pipeline Completed Successfully!")
        return df