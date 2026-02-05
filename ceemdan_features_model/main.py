import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from src.ltsf_linear import NLinear, DLinear
from src.metrics import metric
from src.path import DATA_DIR, SERIES_DIR, CHECKPOINTS_DIR, ROOT_DIR, CACHE_DIR
from src.feature_engineering import create_change_aware_feature
from src.decomposition import run_ceemdan
from plot_visual import plot_all
import warnings


# Disable warnings
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


class IMFFeatureDataset(Dataset):
    """Dataset for Per-IMF training with additional features."""
    def __init__(self, imf_data, features_data, seq_len, pred_len, flag='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = None

        # Combine IMF with features: IMF as first column
        combined_data = np.column_stack([imf_data.reshape(-1, 1), features_data])

        total_len = len(combined_data)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)

        if flag == 'train':
            self.data = combined_data[:train_end]
            # Fit scaler on train data
            self.scaler = StandardScaler()
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)
        elif flag == 'val':
            self.data = combined_data[train_end - seq_len : val_end]
        else:
            self.data = combined_data[val_end - seq_len:]

    def set_scaler(self, scaler):
        """Set scaler and transform data."""
        self.scaler = scaler
        if scaler is not None:
            self.data = scaler.transform(self.data)

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.data[index:s_end]
        seq_y = self.data[s_end:r_end, 0:1]  # Only IMF column as target

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse(self, pred):
        """Inverse transform for IMF predictions (column 0)."""
        if self.scaler is not None:
            dummy = np.zeros((len(pred), self.scaler.n_features_in_))
            dummy[:, 0] = pred
            inv = self.scaler.inverse_transform(dummy)
            return inv[:, 0]
        return pred


def train_imf_model(model_name, pred_len, imf_data, features_data, imf_idx, target):
    """Train a model for a single IMF with additional features."""
    print(f"\n>>> {model_name} | IMF: {imf_idx} | Target: {target}")

    # Create datasets
    train_set = IMFFeatureDataset(imf_data, features_data, BASE_CONFIG['seq_len'], pred_len, flag='train')
    val_set = IMFFeatureDataset(imf_data, features_data, BASE_CONFIG['seq_len'], pred_len, flag='val')
    test_set = IMFFeatureDataset(imf_data, features_data, BASE_CONFIG['seq_len'], pred_len, flag='test')

    # Share scaler from train set
    val_set.set_scaler(train_set.scaler)
    test_set.set_scaler(train_set.scaler)

    train_loader = DataLoader(train_set, batch_size=BASE_CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BASE_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Input channels = 1 (IMF) + features
    in_channels = 1 + features_data.shape[1]

    if model_name == 'DLinear':
        model = DLinear(BASE_CONFIG['seq_len'], pred_len, in_channels).to(device)
    else:
        model = NLinear(BASE_CONFIG['seq_len'], pred_len, in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_CONFIG['lr'])
    criterion = nn.MSELoss()

    best_path = CHECKPOINTS_DIR / f"USGs/Features_{target}/{model_name}_P{pred_len}_IMF{imf_idx}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')

    if not best_path.exists():
        for epoch in range(BASE_CONFIG['epochs']):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_losses.append(criterion(out, by).item())

            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)
    else:
        print(f'>>> Found existing model, loading predictions...')

    # Load best model and predict
    model.load_state_dict(torch.load(best_path))
    model.eval()

    preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            out = model(bx)
            preds.append(test_set.inverse(out.detach().cpu().numpy().squeeze()))

    return np.array(preds), test_set.scaler


def train_and_evaluate(model_name, pred_len, imfs, features_data, target):
    """Train and evaluate model using Per-IMF approach with features."""
    all_imf_preds = []
    all_scalers = []
    print(f"\n>>> Experiment: {model_name} | Target: {target} | Horizon: {pred_len}h | Per-IMF Training")

    # Train each IMF separately
    for i in range(imfs.shape[0]):
        imf_data = imfs[i]
        preds, scaler = train_imf_model(model_name, pred_len, imf_data, features_data, i, target)
        all_imf_preds.append(preds)
        all_scalers.append(scaler)

    # Reconstruct: Sum all IMF predictions
    final_preds = np.sum(all_imf_preds, axis=0)

    # Get Ground Truth by summing all IMF actuals
    final_trues = np.zeros_like(final_preds)
    for i in range(imfs.shape[0]):
        imf_data = imfs[i]
        test_set = IMFFeatureDataset(imf_data, features_data, BASE_CONFIG['seq_len'], pred_len, flag='test')
        test_set.set_scaler(all_scalers[i])

        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        imf_trues = [test_set.inverse(by.numpy().squeeze()) for _, by in test_loader]

        if i == 0:
            final_trues = np.array(imf_trues)
        else:
            ml = min(len(final_trues), len(imf_trues))
            final_trues = final_trues[:ml] + np.array(imf_trues)[:ml]

    # Ensure same length
    ml = min(len(final_preds), len(final_trues))
    final_preds = final_preds[:ml]
    final_trues = final_trues[:ml]

    # Flatten for metrics
    preds_flat = final_preds.flatten()
    trues_flat = final_trues.flatten()
    m = metric(preds_flat, trues_flat)

    # Visual series (last timestep)
    if final_preds.ndim == 2:
        preds_visual = final_preds[:, -1].flatten()
        trues_visual = final_trues[:, -1].flatten()
    else:
        preds_visual = final_preds.flatten()
        trues_visual = final_trues.flatten()

    return {
        'RMSE': m[2], 'MAE': m[0], 'MAPE': m[1], 'R2': m[3],
        'preds_series': preds_visual,
        'trues_series': trues_visual
    }


def main():
    print(f"Training Device: {device}")
    print("=" * 60)
    print("CEEMDAN Features Model - Per-IMF Training")
    print("=" * 60)

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
            imfs = df_imfs.values.T  # Shape: [n_imfs, n_samples]
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

        # Align IMFs with features
        len_diff = imfs.shape[1] - len(df_features)
        if len_diff > 0:
            imfs = imfs[:, len_diff:]  # Trim IMFs to match features length

        features_data = df_features.values

        print(f"IMFs Shape: {imfs.shape}")
        print(f"Features Shape: {features_data.shape}")

        results_summary = []

        for m_name in MODELS:
            for p_len in PRED_LENS:
                res = train_and_evaluate(m_name, p_len, imfs, features_data, target)

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
    print(">>> Per-IMF Training: ENABLED")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SERIES_DIR, exist_ok=True)
    main()
