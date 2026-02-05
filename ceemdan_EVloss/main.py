import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import warnings
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from src.decomposition import run_ceemdan
from src.ltsf_linear import NLinear, DLinear 
from src.feature_engineering import create_change_aware_features
from src.metrics import metric, EventWeightedLoss , EarlyStopping
from src.path import DATA_DIR, SERIES_DIR, CHECKPOINTS_DIR, CACHE_DIR

try:
    from plot_visual import plot_all
except ImportError:
    plot_all = None

warnings.filterwarnings("ignore")

BASE_CONFIG = {
    'seq_len': 168,
    'batch_size': 64,
    'lr': 0.0001,
    'epochs': 100,
    'window_size': 24,
    'percentile': 0.95,
    'ceemd_trials': 50,
    'n_imfs': 12,
    'event_weight': 3.0,
    'patience': 10
}

# Use only EC and pH features from USGS data
TARGETS = ['EC', 'pH']
PRED_LENS = [6, 12, 24, 48, 96, 168]
MODELS = ['DLinear', 'NLinear']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, flag='train', flag_col_index=-1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag_col_index = flag_col_index # Lưu vị trí cột Flag
        
        total_len = len(data)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)
        
        if flag == 'train':
            self.data = data[:train_end]
        elif flag == 'val':
            self.data = data[train_end - seq_len : val_end]
        else:
            self.data = data[val_end - seq_len:]
            
    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        
        seq_x = self.data[index:s_end] 
        
        target = self.data[s_end:r_end, 0:1]
        
        flag   = self.data[s_end:r_end, self.flag_col_index:self.flag_col_index+1]
        
        seq_y = np.concatenate([target, flag], axis=1)
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_target(self, pred, scaler):
        dummy = np.zeros((len(pred), scaler.n_features_in_))
        dummy[:, 0] = pred 
        inv = scaler.inverse_transform(dummy)
        return inv[:, 0]

def train_and_evaluate(model_name, pred_len, df_processed, scaler, target):
    """Train and evaluate model for a specific target variable."""
    in_channels = df_processed.shape[1]

    try:
        flag_idx = df_processed.columns.get_loc('Event_Flag')
    except KeyError:
        flag_idx = -1
        print("Warning: Event_Flag not found, using last column.")

    print(f"\n>>> Model: {model_name} | Target: {target} | Horizon: {pred_len}h | In_Channels: {in_channels}")
    
    data_values = df_processed.values
    train_set = FeatureDataset(data_values, BASE_CONFIG['seq_len'], pred_len, 'train', flag_idx)
    val_set   = FeatureDataset(data_values, BASE_CONFIG['seq_len'], pred_len, 'val', flag_idx)
    test_set  = FeatureDataset(data_values, BASE_CONFIG['seq_len'], pred_len, 'test', flag_idx)
    
    train_loader = DataLoader(train_set, batch_size=BASE_CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=BASE_CONFIG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)
    
    if model_name == 'DLinear':
        model = DLinear(BASE_CONFIG['seq_len'], pred_len, in_channels).to(device)
    else:
        model = NLinear(BASE_CONFIG['seq_len'], pred_len, in_channels).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_CONFIG['lr'])
    
    criterion = EventWeightedLoss(event_weight=BASE_CONFIG['event_weight'])
    

    best_path = CHECKPOINTS_DIR / f"USGs/EventLoss_{target}/{model_name}_P{pred_len}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    
    early_stopping = EarlyStopping(
        patience=BASE_CONFIG['patience'], 
        verbose=True, 
        path=str(best_path) # Chuyển Path sang string cho an toàn
    )
    if not best_path.exists():
        best_loss = float('inf')
        for epoch in range(BASE_CONFIG['epochs']):
            model.train()
            train_losses = []
            
            for bx, by in train_loader:
                bx = bx.to(device)
                
                target_true = by[:, :, 0:1].to(device) 
                event_flag  = by[:, :, 1:2].to(device) 
                
                optimizer.zero_grad()
                out = model(bx)
                
                loss = criterion(out, target_true, event_flag)
                
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    target_true = by[:, :, 0:1].to(device)
                    event_flag  = by[:, :, 1:2].to(device)

                    out = model(bx)
                    v_loss = criterion(out, target_true, event_flag) 
                    val_losses.append(v_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop: 
                print(">>> Early stopping kích hoạt! Dừng huấn luyện.")
                break
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)

    # --- TESTING ---
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    preds_list, trues_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            target_true = by[:, :, 0:1] 
            
            out = model(bx)
            
            # Inverse Transform
            pred_inv = test_set.inverse_target(out.detach().cpu().numpy().flatten(), scaler)
            true_inv = test_set.inverse_target(target_true.detach().cpu().numpy().flatten(), scaler)
            
            preds_list.append(pred_inv)
            trues_list.append(true_inv)

    final_preds = np.array(preds_list)
    final_trues = np.array(trues_list)
    
    # Metric
    m = metric(final_preds.flatten(), final_trues.flatten())
    
    return {
        'RMSE': m[2], 'MAE': m[0], 'R2': m[3], 'MAPE': m[1],
        'preds_series': final_preds[:, -1], 
        'trues_series': final_trues[:, -1]
    }

# --- MAIN DRIVER ---
def main():
    print(f"Device: {device}")
    csv_path = DATA_DIR / 'USGs/water_data_2021_2025_clean.csv'

    if not csv_path.exists():
        print(f"Error: Data file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df[df['site_no'] == 1463500]

    all_results = []

    # Process each target (EC and pH)
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f">>> Processing Target: {target}")
        print(f"{'='*60}")

        target_values = df[target].values

        # Cache IMFs
        imf_save_path = CACHE_DIR / f"imfs_data_usgs_{target}.csv"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if imf_save_path.exists():
            print(f">>> Loading cached IMFs for {target}...")
            df_imfs = pd.read_csv(imf_save_path)
        else:
            print(f">>> Running CEEMDAN for {target}...")
            imfs = run_ceemdan(target_values, trials=BASE_CONFIG['ceemd_trials'], max_imfs=BASE_CONFIG['n_imfs'])
            df_imfs = pd.DataFrame(imfs.T, columns=[f'IMF_{i}' for i in range(imfs.shape[0])])
            df_imfs.to_csv(imf_save_path, index=False)

        # Features + Event Flag
        print(f">>> Creating Change-aware Features & Event Flags for {target}...")
        df_features = create_change_aware_features(
            df,
            target,
            BASE_CONFIG['window_size'],
            BASE_CONFIG['percentile']
        )

        # Alignment & Concat
        print(">>> Aligning Data...")
        len_diff = len(df_imfs) - len(df_features)
        if len_diff > 0:
            df_imfs_trimmed = df_imfs.iloc[len_diff:].reset_index(drop=True)
            df_features = df_features.reset_index(drop=True)
            df_processed = pd.concat([df_features, df_imfs_trimmed], axis=1)
        else:
            df_processed = pd.concat([df_features.reset_index(drop=True), df_imfs.reset_index(drop=True)], axis=1)

        print(f"Final Input Shape: {df_processed.shape}")
        print(f"Columns: {df_processed.columns.tolist()}")

        train_size = int(len(df_processed) * 0.6)
        val_size = int(len(df_processed) * 0.2)

        train_data = df_processed.iloc[:train_size]
        val_data = df_processed.iloc[train_size:train_size + val_size]
        test_data = df_processed.iloc[train_size + val_size:]

        scaler = StandardScaler()
        scaler.fit(train_data.values)

        scaled_values = np.concatenate([
            scaler.transform(train_data.values),
            scaler.transform(val_data.values),
            scaler.transform(test_data.values)
        ], axis=0)
        df_scaled = pd.DataFrame(scaled_values, columns=df_processed.columns)

        results_summary = []
        for m_name in MODELS:
            for p_len in PRED_LENS:
                res = train_and_evaluate(m_name, p_len, df_scaled, scaler, target)

                results_summary.append({
                    'Target': target,
                    'Model': m_name,
                    'Horizon': p_len,
                    'RMSE': res['RMSE'],
                    'MAE': res['MAE'],
                    'R2': res['R2'],
                    'MAPE': res['MAPE']
                })

                series_df = pd.DataFrame({'Actual': res['trues_series'], 'Predicted': res['preds_series']})
                save_path = SERIES_DIR / f"USGs/EventLoss/series_{m_name}_P{p_len}_{target}.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                series_df.to_csv(save_path, index=False)

                print(f"--> Done {m_name} {p_len}h {target}: R2={res['R2']:.4f}")

        # Save results for this target
        final_res_path = DATA_DIR / f"USGs/final_results_eventloss_{target}.csv"
        pd.DataFrame(results_summary).to_csv(final_res_path, index=False)
        all_results.extend(results_summary)

        if plot_all:
            try:
                plot_all(target)
            except Exception as e:
                print(f"Plot error: {e}")

    # Save combined results
    pd.DataFrame(all_results).to_csv(DATA_DIR / "USGs/final_results_eventloss_all.csv", index=False)

    print("\n" + "=" * 60)
    print(">>> ALL EXPERIMENTS COMPLETED!")
    print(f">>> Targets processed: {TARGETS}")
    print("=" * 60)


if __name__ == "__main__":
    main()