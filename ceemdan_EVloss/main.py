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
from src.metrics import metric, EventWeightedLoss, EarlyStopping
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


class IMFEventDataset(Dataset):
    """Dataset for Per-IMF training with event flags."""
    def __init__(self, imf_data, features_data, event_flags, seq_len, pred_len, flag='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = None

        # Combine IMF with features and event flags: IMF as first column, event_flag as last
        combined_data = np.column_stack([imf_data.reshape(-1, 1), features_data, event_flags.reshape(-1, 1)])

        total_len = len(combined_data)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)

        if flag == 'train':
            self.data = combined_data[:train_end]
            # Fit scaler on train data (excluding event_flag column)
            self.scaler = StandardScaler()
            data_to_scale = self.data[:, :-1]  # Exclude event flag
            self.scaler.fit(data_to_scale)
            scaled_data = self.scaler.transform(data_to_scale)
            self.data = np.column_stack([scaled_data, self.data[:, -1:]])  # Add event flag back
        elif flag == 'val':
            self.data = combined_data[train_end - seq_len : val_end]
        else:
            self.data = combined_data[val_end - seq_len:]

    def set_scaler(self, scaler):
        """Set scaler and transform data."""
        self.scaler = scaler
        if scaler is not None:
            data_to_scale = self.data[:, :-1]  # Exclude event flag
            scaled_data = scaler.transform(data_to_scale)
            self.data = np.column_stack([scaled_data, self.data[:, -1:]])  # Add event flag back

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.data[index:s_end]

        # Target: IMF (column 0), Event flag: last column
        target = self.data[s_end:r_end, 0:1]
        event_flag = self.data[s_end:r_end, -1:]

        seq_y = np.concatenate([target, event_flag], axis=1)

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


def train_imf_model(model_name, pred_len, imf_data, features_data, event_flags, imf_idx, target):
    """Train a model for a single IMF with event-weighted loss."""
    print(f"\n>>> {model_name} | IMF: {imf_idx} | Target: {target}")

    # Create datasets
    train_set = IMFEventDataset(imf_data, features_data, event_flags, BASE_CONFIG['seq_len'], pred_len, flag='train')
    val_set = IMFEventDataset(imf_data, features_data, event_flags, BASE_CONFIG['seq_len'], pred_len, flag='val')
    test_set = IMFEventDataset(imf_data, features_data, event_flags, BASE_CONFIG['seq_len'], pred_len, flag='test')

    # Share scaler from train set
    val_set.set_scaler(train_set.scaler)
    test_set.set_scaler(train_set.scaler)

    train_loader = DataLoader(train_set, batch_size=BASE_CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BASE_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Input channels = 1 (IMF) + features + 1 (event_flag)
    in_channels = 1 + features_data.shape[1] + 1

    if model_name == 'DLinear':
        model = DLinear(BASE_CONFIG['seq_len'], pred_len, in_channels).to(device)
    else:
        model = NLinear(BASE_CONFIG['seq_len'], pred_len, in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_CONFIG['lr'])
    criterion = EventWeightedLoss(event_weight=BASE_CONFIG['event_weight'])

    best_path = CHECKPOINTS_DIR / f"USGs/EventLoss_{target}/{model_name}_P{pred_len}_IMF{imf_idx}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=BASE_CONFIG['patience'], verbose=False, path=str(best_path))

    if not best_path.exists():
        best_loss = float('inf')
        for epoch in range(BASE_CONFIG['epochs']):
            model.train()
            for bx, by in train_loader:
                bx = bx.to(device)
                target_true = by[:, :, 0:1].to(device)
                event_flag = by[:, :, 1:2].to(device)

                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, target_true, event_flag)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    target_true = by[:, :, 0:1].to(device)
                    event_flag = by[:, :, 1:2].to(device)
                    out = model(bx)
                    val_losses.append(criterion(out, target_true, event_flag).item())

            avg_val_loss = np.mean(val_losses)
            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop:
                print(f">>> Early stopping at epoch {epoch}")
                break

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


def train_and_evaluate(model_name, pred_len, imfs, features_data, event_flags, target):
    """Train and evaluate model using Per-IMF approach with event-weighted loss."""
    all_imf_preds = []
    all_scalers = []
    print(f"\n>>> Experiment: {model_name} | Target: {target} | Horizon: {pred_len}h | Per-IMF Training + EventLoss")

    # Train each IMF separately
    for i in range(imfs.shape[0]):
        imf_data = imfs[i]
        preds, scaler = train_imf_model(model_name, pred_len, imf_data, features_data, event_flags, i, target)
        all_imf_preds.append(preds)
        all_scalers.append(scaler)

    # Reconstruct: Sum all IMF predictions
    final_preds = np.sum(all_imf_preds, axis=0)

    # Get Ground Truth by summing all IMF actuals
    final_trues = np.zeros_like(final_preds)
    for i in range(imfs.shape[0]):
        imf_data = imfs[i]
        test_set = IMFEventDataset(imf_data, features_data, event_flags, BASE_CONFIG['seq_len'], pred_len, flag='test')
        test_set.set_scaler(all_scalers[i])

        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        imf_trues = [test_set.inverse(by[:, :, 0].numpy().squeeze()) for _, by in test_loader]

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


# --- MAIN DRIVER ---
def main():
    print(f"Device: {device}")
    print("=" * 60)
    print("CEEMDAN EventLoss Model - Per-IMF Training")
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

        target_values = df[target].values

        # Cache IMFs
        imf_save_path = CACHE_DIR / f"imfs_data_usgs_{target}.csv"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if imf_save_path.exists():
            print(f">>> Loading cached IMFs for {target}...")
            df_imfs = pd.read_csv(imf_save_path)
            imfs = df_imfs.values.T  # Shape: [n_imfs, n_samples]
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

        # Align IMFs with features
        len_diff = imfs.shape[1] - len(df_features)
        if len_diff > 0:
            imfs = imfs[:, len_diff:]  # Trim IMFs to match features length

        # Extract event flags (should be in df_features)
        if 'Event_Flag' in df_features.columns:
            event_flags = df_features['Event_Flag'].values
            features_data = df_features.drop(columns=['Event_Flag']).values
        else:
            print("Warning: Event_Flag not found, creating from target changes...")
            changes = np.abs(np.diff(target_values, prepend=target_values[0]))
            threshold = np.percentile(changes, BASE_CONFIG['percentile'] * 100)
            event_flags = (changes > threshold).astype(float)
            event_flags = event_flags[len_diff:] if len_diff > 0 else event_flags
            features_data = df_features.values

        print(f"IMFs Shape: {imfs.shape}")
        print(f"Features Shape: {features_data.shape}")
        print(f"Event Flags Shape: {event_flags.shape}")

        results_summary = []

        for m_name in MODELS:
            for p_len in PRED_LENS:
                res = train_and_evaluate(m_name, p_len, imfs, features_data, event_flags, target)

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
    print(">>> Per-IMF Training: ENABLED")
    print(">>> Event-Weighted Loss: ENABLED")
    print("=" * 60)


if __name__ == "__main__":
    main()
