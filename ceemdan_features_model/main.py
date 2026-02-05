import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from src.ltsf_linear import NLinear, DLinear
from src.metrics import metric
from src.path import DATA_DIR, SERIES_DIR, CHECKPOINTS_DIR,ROOT_DIR,CACHE_DIR
from src.data_loader import FeatureDataset
from src.feature_engineering import create_change_aware_feature
from src.decomposition import run_ceemdan
from plot_visual import plot_all
import warnings


# Tắt cảnh báo
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_CONFIG = {
    'seq_len': 168,
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 50,
    'ceemd_trials': 50,
    'n_imfs': 12,
    'window_size': 24,
    'percentile': 0.95
}

# Use only EC and pH features from USGS data
TARGETS = ['EC', 'pH']
PRED_LENS = [6, 12, 24, 48, 96, 168]
MODELS = ['DLinear', 'NLinear']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_evaluate(model_name, pred_len, df_processed, scaler, target):
    """Train and evaluate model for a specific target variable."""
    in_channels = df_processed.shape[1]
    print(f"\n>>> Experiment: {model_name} | Target: {target} | Horizon: {pred_len}h | Input Channels: {in_channels}")
    
    # Tạo Dataset
    data_values = df_processed.values
    train_set = FeatureDataset(data_values, BASE_CONFIG['seq_len'], pred_len, flag='train')
    val_set   = FeatureDataset(data_values, BASE_CONFIG['seq_len'], pred_len, flag='val')
    test_set  = FeatureDataset(data_values, BASE_CONFIG['seq_len'], pred_len, flag='test')
    
    train_loader = DataLoader(train_set, batch_size=BASE_CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=BASE_CONFIG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)
    

    if model_name == 'DLinear':
        model = DLinear(BASE_CONFIG['seq_len'], pred_len,in_channels).to(device)
    else:
        model = NLinear(BASE_CONFIG['seq_len'], pred_len,in_channels).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_CONFIG['lr'])
    criterion = nn.MSELoss()
    
    best_path = CHECKPOINTS_DIR / f"USGs/Features_{target}/{model_name}_P{pred_len}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')

    if not best_path.exists():
        for epoch in range(BASE_CONFIG['epochs']):
            model.train()
            train_losses = []
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                
                out = model(bx)
                loss = criterion(out, by)
                
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    loss = criterion(out, by)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)
                print(f"Epoch {epoch}: Val Loss {avg_val_loss:.5f} -> Saved")

    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    preds_list = []
    trues_list = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)

            pred_inv = test_set.inverse_target(out[:, :, 0].detach().cpu().numpy().squeeze(0), scaler)
            true_inv = test_set.inverse_target(by[:, :, 0].detach().cpu().numpy().squeeze(0), scaler)
            
            preds_list.append(pred_inv)
            trues_list.append(true_inv)

    final_preds = np.array(preds_list) 
    final_trues = np.array(trues_list)
    
    preds_flat = final_preds.flatten()
    trues_flat = final_trues.flatten()
    m = metric(preds_flat, trues_flat)
    
    preds_visual = final_preds[:, -1].flatten()
    trues_visual = final_trues[:, -1].flatten()
    
    return {
        'RMSE': m[2], 'MAE': m[0], 'MAPE': m[1], 'R2': m[3], 
        'preds_series': preds_visual, 
        'trues_series': trues_visual
    }

def main():
    print(f"Training Device: {device}")

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

        target_data = df[target].values

        # Cache IMFs
        imf_save_path = CACHE_DIR / f"imfs_data_usgs_{target}.csv"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if imf_save_path.exists():
            print(f">>> Loading cached IMFs for {target}...")
            df_imfs = pd.read_csv(imf_save_path)
        else:
            print(f">>> Running CEEMDAN for {target}...")
            imfs = run_ceemdan(target_data, trials=BASE_CONFIG['ceemd_trials'], max_imfs=BASE_CONFIG['n_imfs'])
            df_imfs = pd.DataFrame(imfs.T, columns=[f'IMF_{i}' for i in range(imfs.shape[0])])
            df_imfs.to_csv(imf_save_path, index=False)

        print(f">>> Creating Rolling Features for {target}...")
        df_features = create_change_aware_feature(
            df,
            target_col=target,
            window_size=BASE_CONFIG['window_size'],
            percentile=BASE_CONFIG['percentile']
        )

        len_diff = len(df_imfs) - len(df_features)
        if len_diff > 0:
            df_imfs_trimmed = df_imfs.iloc[len_diff:].reset_index(drop=True)
            df_features = df_features.reset_index(drop=True)
            df_processed = pd.concat([df_features, df_imfs_trimmed], axis=1)
        else:
            df_processed = pd.concat([df_features.reset_index(drop=True), df_imfs.reset_index(drop=True)], axis=1)

        print(f"Feature Columns: {df_features.columns.tolist()}")
        print(f"IMF Columns: {df_imfs.columns.tolist()}")
        print(f"Final Data Shape: {df_processed.shape}")

        train_size = int(len(df_processed) * 0.7)
        val_size = int(len(df_processed) * 0.1)

        train_data = df_processed.iloc[:train_size]
        val_data = df_processed.iloc[train_size:train_size + val_size]
        test_data = df_processed.iloc[train_size + val_size:]

        scaler = StandardScaler()
        scaler.fit(train_data.values)

        train_scaled = scaler.transform(train_data.values)
        val_scaled = scaler.transform(val_data.values)
        test_scaled = scaler.transform(test_data.values)

        scaled_values = np.concatenate([train_scaled, val_scaled, test_scaled], axis=0)
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
                    'MAPE': res['MAPE'],
                    'R2': res['R2']
                })

                series_df = pd.DataFrame({'Actual': res['trues_series'], 'Predicted': res['preds_series']})
                save_path = SERIES_DIR / f"USGs/series_{m_name}_P{p_len}_{target}_feat.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                series_df.to_csv(save_path, index=False)

                print(f"--> Done {m_name} {p_len}h {target}: R2={res['R2']:.4f}")

        # Save results for this target
        final_res_path = DATA_DIR / f"USGs/final_results_{target}_features.csv"
        pd.DataFrame(results_summary).to_csv(final_res_path, index=False)
        all_results.extend(results_summary)

        print(f">>> Generating plots for {target}...")
        try:
            plot_all(target)
        except Exception as e:
            print(f"Plot error: {e}")

    # Save combined results
    pd.DataFrame(all_results).to_csv(DATA_DIR / "USGs/final_results_features_all.csv", index=False)

    print("\n" + "=" * 60)
    print(">>> ALL EXPERIMENTS COMPLETED!")
    print(f">>> Targets processed: {TARGETS}")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SERIES_DIR, exist_ok=True)
    main()
