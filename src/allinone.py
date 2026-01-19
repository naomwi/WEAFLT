import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from PyEMD import CEEMDAN
import warnings
from typing import Dict, List
import time
import random

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Running on device: {device}")

# Hyperparameters
CONFIG = {
    'seq_len': 96,
    'pred_len': 24,
    'batch_size': 32,
    'learning_rate': 0.005,
    'epochs': 15,
    'event_weight': 3.0,
    'event_threshold_pct': 95,
}

# Runtime & Stability Tracking
RUNTIME_LOG = []
STABILITY_LOG = []

def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ==================================================================================
# 1. DATA LOADING & PREPROCESSING (v3.3.2 Fixed)
# ==================================================================================

def detect_events_from_threshold(df, threshold):
    """Event detection using raw log-diff from training distribution"""
    df["is_event"] = (df["abs_delta"] > threshold).astype(float)
    return df

def preprocess_dataframe(df, col_name="OT", event_threshold=None, compute_rolling=False):
    """Data preprocessing with proper NaN handling"""
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if col_name != "OT" and col_name in df.columns:
        df = df.rename(columns={col_name: "OT"})

    df["OT"] = pd.to_numeric(df["OT"], errors='coerce')

    if df["OT"].isna().sum() > 0:
        df["OT"] = df["OT"].fillna(method='ffill').fillna(method='bfill')

    if df["OT"].isna().sum() > 0:
        mean_val = df["OT"].mean()
        if pd.isna(mean_val): mean_val = 1.0
        df["OT"] = df["OT"].fillna(mean_val)

    df["OT"] = df["OT"].clip(lower=0.1)
    df["OT_log"] = np.log(df["OT"] + 1e-6)
    df["delta_x"] = df["OT_log"].diff().fillna(0)
    df["abs_delta"] = df["delta_x"].abs()

    if event_threshold is not None:
        df = detect_events_from_threshold(df, event_threshold)
    else:
        df["is_event"] = 0.0

    if compute_rolling:
        df["rolling_std"] = df["OT_log"].rolling(12).std().fillna(0)
        df["rolling_zscore"] = ((df["OT_log"] - df["OT_log"].rolling(24).mean()) /
                                (df["OT_log"].rolling(24).std() + 1e-6)).fillna(0)
    else:
        df["rolling_std"] = 0.0
        df["rolling_zscore"] = 0.0

    df["relative_change"] = df["OT"].pct_change().fillna(0)
    df["ma20"] = df["relative_change"].rolling(20).mean().fillna(0)

    return df

def compute_rolling_features_post_split(df, train_end_idx):
    """Compute rolling features without leakage"""
    df_copy = df.copy()
    signal = df_copy["OT_log"].values

    rolling_std = np.zeros(len(signal))
    rolling_zscore = np.zeros(len(signal))

    for i in range(len(signal)):
        if i < 12:
            rolling_std[i] = 0
        else:
            if i <= train_end_idx:
                window = signal[max(0, i-12):i]
            else:
                window = signal[max(0, i-12):min(i, train_end_idx+1)]
            rolling_std[i] = np.std(window) if len(window) > 0 else 0

        if i < 24:
            rolling_zscore[i] = 0
        else:
            if i <= train_end_idx:
                window = signal[max(0, i-24):i]
            else:
                window = signal[max(0, i-24):min(i, train_end_idx+1)]

            if len(window) > 0:
                mean = np.mean(window)
                std = np.std(window) + 1e-6
                rolling_zscore[i] = (signal[i] - mean) / std
            else:
                rolling_zscore[i] = 0

    df_copy["rolling_std"] = rolling_std
    df_copy["rolling_zscore"] = rolling_zscore

    return df_copy

def compute_event_threshold_from_train(df_train, percentile=95):
    """Compute event threshold from training data only"""
    threshold = np.percentile(df_train["abs_delta"], percentile)
    return threshold

def load_data_source_separate():
    """Load from separate files"""
    print("\n📂 [1/8] Loading separate files...")
    try:
        df_ec = pd.read_csv("/content/drive/MyDrive/Thư mục không có tiêu đề/WEATHER_/datasets/G_WTP-main/G_WTP-main/EC_origin.csv")
        df_ph = pd.read_csv("/content/drive/MyDrive/Thư mục không có tiêu đề/WEATHER_/datasets/G_WTP-main/G_WTP-main/pH_origin.csv")
        return preprocess_dataframe(df_ec, "OT"), preprocess_dataframe(df_ph, "OT")
    except FileNotFoundError:
        print("⚠️ Error: Missing EC_origin.csv or pH_origin.csv")
        return None, None

def load_data_source_api():
    """Load from API file"""
    print("\n📂 [1/8] Loading API file...")
    try:
        df = pd.read_csv("/content/drive/MyDrive/Thư mục không có tiêu đề/WEATHER_/datasets/water_data_api/water_data_api.csv")
        col_date = "DateTime" if "DateTime" in df.columns else "date"
        df["date"] = pd.to_datetime(df[col_date])
        df_ec = df[["date", "EC"]].copy()
        df_ph = df[["date", "pH"]].copy()
        return preprocess_dataframe(df_ec, "EC"), preprocess_dataframe(df_ph, "pH")
    except FileNotFoundError:
        print("⚠️ Error: Missing water_data_api.csv")
        return None, None

# ==================================================================================
# 2. 🔥 CEEMD DECOMPOSITION - FULL POWER VERSION
# ==================================================================================

def apply_ceemd_decomposition(df, target_col="OT_log", n_imfs=5):
    """
    🔥 FULL POWER CEEMD decomposition

    Changes from limited version:
    - NO sample limit (use ALL data)
    - trials: 25 → 100 (4x more robust)
    - noise_scale: 0.1 → 0.2 (better decomposition)
    - parallel: False → True (multi-core processing)
    - max_imf: 3 → 5 (more frequency details)

    Runtime: ~10-15 minutes (worth it for thesis quality)
    """
    print(f"\n🌊 [2/8] Applying CEEMD (FULL POWER MODE)...")

    signal_full = df[target_col].values
    n_total = len(signal_full)

    # 🔥 CHANGE 1: Use ALL data (no limit)
    print(f"   📊 Processing ALL {n_total} samples (Full Power)")
    signal = signal_full
    start_idx = 0

    try:
        # 🔥 CHANGE 2: Full Power parameters
        ceemdan = CEEMDAN(
            trials=100,           # 4x more robust
            noise_scale=0.2,      # Better decomposition
            parallel=True         # Multi-core processing
        )

        print(f"   ⚙️ Config: trials=100, noise=0.2, parallel=True")
        print(f"   ⏱️ Estimated time: 10-15 minutes...")

        start_time = time.time()

        # 🔥 CHANGE 3: 5 IMFs (more frequency details)
        ceemdan.ceemdan(signal, max_imf=n_imfs)
        imfs, residue = ceemdan.get_imfs_and_residue()

        elapsed = time.time() - start_time
        print(f"   ⏱️ CEEMD completed in {elapsed:.1f}s ({elapsed/60:.2f} min)")

        n_imfs_actual = min(n_imfs, len(imfs))

        for i in range(n_imfs_actual):
            if start_idx > 0:
                padded = np.concatenate([np.zeros(start_idx), imfs[i]])
            else:
                padded = imfs[i]
            df[f"IMF_{i}"] = padded

        if start_idx > 0:
            df["residue"] = np.concatenate([np.zeros(start_idx), residue])
        else:
            df["residue"] = residue

        print(f"   ✅ CEEMDAN success: {n_imfs_actual} IMFs (Full Power)")

        # Print variance contributions
        variances = [df[f"IMF_{i}"].var() for i in range(n_imfs_actual)]
        variances.append(df["residue"].var())
        total_var = sum(variances)

        print(f"   📊 Variance contributions:")
        for i in range(n_imfs_actual):
            pct = variances[i] / total_var * 100
            print(f"      IMF_{i}: {pct:.1f}%")
        print(f"      Residue: {variances[-1] / total_var * 100:.1f}%")

    except Exception as e:
        print(f"   ⚠️ CEEMDAN failed: {e}")
        print(f"   🔄 Using fallback method...")

        # Fallback method
        for i in range(min(n_imfs, 3)):
            window = [10, 50, 200][i]
            ma = pd.Series(signal_full).rolling(window, center=True).mean()
            ma = ma.fillna(method='ffill').fillna(method='bfill')
            df[f"IMF_{i}"] = signal_full - ma.values

        residue_ma = pd.Series(signal_full).rolling(200, center=True).mean()
        df["residue"] = residue_ma.fillna(method='ffill').fillna(method='bfill').values
        n_imfs_actual = min(n_imfs, 3)

    return df, n_imfs_actual


# ==================================================================================
# 3. MODELS
# ==================================================================================

class NLinear(nn.Module):
    """Baseline NLinear model"""
    def __init__(self, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = x.squeeze(-1)
        last_value = x[:, -1:]
        x_norm = x - last_value
        pred_norm = self.linear(x_norm)
        pred = pred_norm + last_value
        return pred.unsqueeze(-1)

class DLinear(nn.Module):
    """DLinear model"""
    def __init__(self, seq_len, pred_len):
        super(DLinear, self).__init__()
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.kernel_size = 25
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def decompose(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal

    def forward(self, x):
        trend, seasonal = self.decompose(x)
        trend_pred = self.linear_trend(trend.squeeze(-1))
        seasonal_pred = self.linear_seasonal(seasonal.squeeze(-1))
        return (trend_pred + seasonal_pred).unsqueeze(-1)

class CALinear(nn.Module):
    """CALinear with multi-feature support"""
    def __init__(self, seq_len, pred_len, n_features=3):
        super(CALinear, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.linear_main = nn.Linear(seq_len, pred_len)
        self.linear_std = nn.Linear(seq_len, pred_len)
        self.linear_zscore = nn.Linear(seq_len, pred_len)
        self.fusion = nn.Linear(3 * pred_len, pred_len)

    def forward(self, x):
        if x.shape[-1] == 1:
            x_main = x[:, :, 0]
            x_std = torch.zeros_like(x_main)
            x_zscore = torch.zeros_like(x_main)
        elif x.shape[-1] >= 3:
            x_main = x[:, :, 0]
            x_std = x[:, :, 1]
            x_zscore = x[:, :, 2]
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        last_val = x_main[:, -1:]
        x_main_norm = x_main - last_val

        pred_main = self.linear_main(x_main_norm)
        pred_std = self.linear_std(x_std)
        pred_zscore = self.linear_zscore(x_zscore)

        concat = torch.cat([pred_main, pred_std, pred_zscore], dim=1)
        pred_fused = self.fusion(concat)
        pred = pred_fused + last_val

        return pred.unsqueeze(-1)

# ==================================================================================
# 4. DATASET & LOSS
# ==================================================================================

class WaterQualityDataset(Dataset):
    """Normalization computed externally from train data only"""
    def __init__(self, data, seq_len, pred_len, use_features=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_features = use_features

        self.series = torch.FloatTensor(data["OT_log"].values).unsqueeze(-1)
        self.events = torch.FloatTensor(data["is_event"].values).unsqueeze(-1)

        if use_features:
            self.std = torch.FloatTensor(data["rolling_std"].values).unsqueeze(-1)
            self.zscore = torch.FloatTensor(data["rolling_zscore"].values).unsqueeze(-1)

        self.mean = None
        self.std_val = None
        self.series_norm = None

    def set_normalization(self, mean, std):
        """Set normalization parameters from external source"""
        self.mean = mean
        self.std_val = std

        if torch.abs(self.std_val) < 1e-6:
            self.std_val = torch.tensor(1.0)

        self.series_norm = (self.series - self.mean) / self.std_val

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_main = self.series_norm[idx : idx + self.seq_len]
        y = self.series_norm[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        y_event = self.events[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        if self.use_features:
            x_std = self.std[idx : idx + self.seq_len]
            x_zscore = self.zscore[idx : idx + self.seq_len]
            x = torch.cat([x_main, x_std, x_zscore], dim=-1)
        else:
            x = x_main

        return x, y, y_event

    def denormalize(self, x):
        return x * self.std_val + self.mean

class EventWeightedMSE(nn.Module):
    """Event-Weighted Loss (used for ALL models in fair comparison)"""
    def __init__(self, alpha=3.0):
        super(EventWeightedMSE, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, is_event):
        loss = self.mse(pred, target)
        weights = torch.ones_like(loss)
        weights[is_event == 1] = self.alpha
        return (loss * weights).mean()

def create_dataloaders(df, seq_len, pred_len, batch_size=32, use_features=False):
    """Compute normalization ONLY from train split"""
    dataset = WaterQualityDataset(df, seq_len, pred_len, use_features=use_features)

    total_len = len(dataset)
    train_size = int(0.6 * total_len)
    val_size = int(0.2 * total_len)

    # Compute mean/std from train indices only
    train_indices = range(0, train_size)
    train_data = dataset.series[train_indices]
    mean = torch.mean(train_data)
    std = torch.std(train_data)

    if torch.abs(std) < 1e-6:
        std = torch.tensor(1.0)

    dataset.set_normalization(mean, std)

    train_set = torch.utils.data.Subset(dataset, range(0, train_size))
    val_set = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_set = torch.utils.data.Subset(dataset, range(train_size + val_size, total_len))

    loaders = (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )

    split_info = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': total_len - train_size - val_size
    }

    return loaders, dataset, split_info

# ==================================================================================
# 5. EVALUATION
# ==================================================================================

def evaluate_model(model, loader, dataset, device):
    """Evaluate model"""
    model.eval()
    preds, actuals, events = [], [], []

    with torch.no_grad():
        for bx, by, bevent in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds.append(dataset.denormalize(out).cpu().numpy())
            actuals.append(dataset.denormalize(by).cpu().numpy())
            events.append(bevent.cpu().numpy())

    if len(preds) == 0:
        return {}

    preds = np.concatenate(preds).flatten()
    actuals = np.concatenate(actuals).flatten()
    events = np.concatenate(events).flatten()

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-6))) * 100

    event_indices = np.where(events == 1)[0]
    if len(event_indices) > 0:
        sf_mae = mean_absolute_error(actuals[event_indices], preds[event_indices])
    else:
        sf_mae = 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "SF_MAE": sf_mae,
        "N_Events": len(event_indices)
    }

# ==================================================================================
# 6. XAI (FIXED BASELINE STRATEGY)
# ==================================================================================

def temporal_attribution_enhanced(model, dataset, input_window, baseline_window,
                                  segment_size=4, device="cpu"):
    """
    🔧 FIXED: TimeSHAP-compliant baseline (mean across ALL features, not just channel 0)
    Temporal attribution using mean-baseline masking.
    Baseline is computed as the temporal mean across all features.
    """
    model.eval()

    if input_window.ndim == 1:
        input_window = input_window.reshape(-1, 1)
    if baseline_window.ndim == 1:
        baseline_window = baseline_window.reshape(-1, 1)

    T = input_window.shape[0]
    attributions_magnitude = []

    x = torch.FloatTensor(input_window).unsqueeze(0).to(device)

    with torch.no_grad():
        y_full = model(x)
        y_full_denorm = dataset.denormalize(y_full).cpu().numpy().flatten()

    for start in range(0, T, segment_size):
        end = min(start + segment_size, T)

        x_masked = input_window.copy()
        x_masked[start:end] = baseline_window[start:end]

        x_m = torch.FloatTensor(x_masked).unsqueeze(0).to(device)

        with torch.no_grad():
            y_mask = model(x_m)
            y_mask_denorm = dataset.denormalize(y_mask).cpu().numpy().flatten()

        diff = y_full_denorm - y_mask_denorm
        magnitude = np.mean(np.abs(diff))
        attributions_magnitude.append(magnitude)

    return np.array(attributions_magnitude)

# ==================================================================================
# 7. TRAINING (WITH RUNTIME TRACKING)
# ==================================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs, device, model_name="Model"):
    """⏱️ Training loop with RUNTIME TRACKING"""
    best_val_loss = float('inf')
    epoch_times = []

    for epoch in range(epochs):
        # ⏱️ START EPOCH TIMER
        epoch_start = time.time()

        model.train()
        train_loss = 0

        for bx, by, bevent in train_loader:
            bx, by, bevent = bx.to(device), by.to(device), bevent.to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by, bevent)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for bx, by, bevent in val_loader:
                bx, by, bevent = bx.to(device), by.to(device), bevent.to(device)
                pred = model(bx)
                loss = criterion(pred, by, bevent)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # ⏱️ END EPOCH TIMER
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # ⏱️ LOG AVERAGE TRAINING TIME
    avg_epoch_time = np.mean(epoch_times)
    RUNTIME_LOG.append({
        "Model": model_name,
        "Training_Time": avg_epoch_time * epochs,
        "Inference_Time": 0  # Will be filled later
    })

    return model

def train_model_simple(model, train_loader, val_loader, criterion, optimizer,
                      epochs, device, model_name="Model"):
    """Training loop for standard MSE (with runtime)"""
    best_val_loss = float('inf')
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0

        for bx, by, bevent in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for bx, by, bevent in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    avg_epoch_time = np.mean(epoch_times)
    RUNTIME_LOG.append({
        "Model": model_name,
        "Training_Time": avg_epoch_time * epochs,
        "Inference_Time": 0
    })

    return model


def run_experiment(df, target_name, source_name, model_type="NLinear"):
    """
    🔧 FIXED: Correct inference time calculation (per-sample, not per-batch)
    Run experiment with RUNTIME tracking for Training, Inference, and XAI.
    """
    print(f"\n🚀 Training {model_type} on {source_name} ({target_name})...")

    seq_len, pred_len = CONFIG['seq_len'], CONFIG['pred_len']
    use_features = (model_type == "CALinear")

    # Create temporary dataset for split info only
    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], use_features=False
    )

    train_size = split_info['train_size']
    train_end_idx = train_size + seq_len

    df_train = df.iloc[:train_end_idx].copy()
    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )

    df = detect_events_from_threshold(df, event_threshold)

    if use_features:
        df = compute_rolling_features_post_split(df, train_size + seq_len)

    loaders, dataset, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], use_features=use_features
    )

    train_loader, val_loader, test_loader = loaders

    if model_type == "NLinear":
        model = NLinear(seq_len, pred_len).to(device)
    elif model_type == "DLinear":
        model = DLinear(seq_len, pred_len).to(device)
    elif model_type == "CALinear":
        model = CALinear(seq_len, pred_len, n_features=3).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = EventWeightedMSE(alpha=CONFIG['event_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                       CONFIG['epochs'], device, model_name=model_type)

    # ⏱️ FIXED: MEASURE INFERENCE TIME PER SAMPLE CORRECTLY
    inference_times_per_sample = []
    model.eval()

    with torch.no_grad():
        for bx, by, _ in test_loader:
            batch_size = bx.size(0)
            start = time.time()
            _ = model(bx.to(device))
            elapsed = time.time() - start
            # Divide by batch size to get per-sample time
            inference_times_per_sample.append(elapsed / batch_size)

    # Average inference time per sample in milliseconds
    avg_infer_ms = np.mean(inference_times_per_sample) * 1000

    # Update RUNTIME_LOG
    for entry in RUNTIME_LOG:
        if entry["Model"] == model_type and entry["Inference_Time"] == 0:
            entry["Inference_Time"] = avg_infer_ms
            break

    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({
        "Target": target_name,
        "Source": source_name,
        "Model": model_type
    })

    return metrics, model, dataset, test_loader, df

# ==================================================================================
# 8. EXPERIMENTS
# ==================================================================================

def run_multi_horizon_experiments(df, target_name, source_name, model_type="NLinear",
                                  horizons=[6, 12, 24, 48, 72]):
    """Multi-horizon with runtime tracking"""
    print(f"\n🔬 Multi-horizon: {model_type}...")
    results_by_horizon = []

    original_pred_len = CONFIG['pred_len']

    for pred_len in horizons:
        print(f"   Horizon: {pred_len}h")

        try:
            CONFIG['pred_len'] = pred_len
            metrics, model, dataset, test_loader, df_updated = run_experiment(
                df, target_name, source_name, model_type
            )
            metrics['Horizon'] = pred_len
            results_by_horizon.append(metrics)

        except Exception as e:
            print(f"   ⚠️ Failed: {str(e)[:50]}")

        finally:
            CONFIG['pred_len'] = original_pred_len

    return pd.DataFrame(results_by_horizon)

def run_ablation_study(df, target_name, source_name):
    """Ablation study"""
    print(f"\n🔬 Ablation study...")
    results = []

    seq_len, pred_len = CONFIG['seq_len'], CONFIG['pred_len']

    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], False
    )

    train_size = split_info['train_size']
    train_end_idx = train_size + seq_len

    df_train = df.iloc[:train_end_idx].copy()
    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )

    df = detect_events_from_threshold(df, event_threshold)

    # 1. Baseline
    print("   [1/4] Baseline")
    loaders, dataset, _ = create_dataloaders(df, seq_len, pred_len, CONFIG['batch_size'], False)
    train_loader, val_loader, test_loader = loaders

    model = NLinear(seq_len, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model_simple(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-Baseline")

    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "Baseline", "Change_Features": "No", "Event_Loss": "No"})
    results.append(metrics)

    # 2. + Features
    print("   [2/4] + Features")
    df_with_features = compute_rolling_features_post_split(df, train_size + seq_len)
    loaders, dataset, _ = create_dataloaders(df_with_features, seq_len, pred_len, CONFIG['batch_size'], True)
    train_loader, val_loader, test_loader = loaders

    model = CALinear(seq_len, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model_simple(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-+Features")

    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "+ Features", "Change_Features": "Yes", "Event_Loss": "No"})
    results.append(metrics)

    # 3. + Event loss
    print("   [3/4] + Event loss")
    loaders, dataset, _ = create_dataloaders(df, seq_len, pred_len, CONFIG['batch_size'], False)
    train_loader, val_loader, test_loader = loaders

    model = NLinear(seq_len, pred_len).to(device)
    criterion = EventWeightedMSE(alpha=CONFIG['event_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-+EventLoss")

    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "+ Event loss", "Change_Features": "No", "Event_Loss": "Yes"})
    results.append(metrics)

    # 4. Full
    print("   [4/4] Full")
    loaders, dataset, _ = create_dataloaders(df_with_features, seq_len, pred_len, CONFIG['batch_size'], True)
    train_loader, val_loader, test_loader = loaders

    model = CALinear(seq_len, pred_len).to(device)
    criterion = EventWeightedMSE(alpha=CONFIG['event_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-Full")

    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "Full", "Change_Features": "Yes", "Event_Loss": "Yes"})
    results.append(metrics)

    return pd.DataFrame(results)

def run_alpha_sensitivity_analysis(df, target_name, source_name,
                                   alphas=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0]):
    """Alpha sensitivity"""
    print(f"\n🔬 Alpha sensitivity...")
    results = []

    seq_len, pred_len = CONFIG['seq_len'], CONFIG['pred_len']

    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], False
    )

    train_size = split_info['train_size']
    train_end_idx = train_size + seq_len

    df_train = df.iloc[:train_end_idx].copy()
    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )

    df = detect_events_from_threshold(df, event_threshold)
    df = compute_rolling_features_post_split(df, train_size + seq_len)

    for alpha in alphas:
        print(f"   Alpha={alpha}")

        loaders, dataset, _ = create_dataloaders(df, seq_len, pred_len, CONFIG['batch_size'], True)
        train_loader, val_loader, test_loader = loaders

        model = CALinear(seq_len, pred_len).to(device)
        criterion = EventWeightedMSE(alpha=alpha)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        model = train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, f"Alpha-{alpha}")

        metrics = evaluate_model(model, test_loader, dataset, device)
        metrics['Alpha'] = alpha
        results.append(metrics)

    return pd.DataFrame(results)

# ==================================================================================
# 9. 🆕 STABILITY ANALYSIS (FIXED)
# ==================================================================================

def run_stability_analysis(model, dataset, test_loader, device, model_name="Model"):
    """
    🔧 FIXED:
    1. Attribution Variance now measures variance of attribution values, not correlations
    2. Renamed "Seed Consistency" to "Perturbation Consistency"
    3. TimeSHAP-compliant baseline (mean across all features)

    🛡️ COMPREHENSIVE STABILITY ANALYSIS
    """
    print(f"\n🛡️ Running Stability Analysis for {model_name}...")

    # Select multiple samples
    test_indices = list(test_loader.dataset.indices)
    selected_indices = []

    # Try to find event samples
    for idx in test_indices:
        _, _, evt = dataset[idx]
        if evt.sum() > 0:
            selected_indices.append(idx)
            if len(selected_indices) >= 5:
                break

    # Fill with non-event samples if needed
    while len(selected_indices) < 10 and len(selected_indices) < len(test_indices):
        idx = test_indices[len(selected_indices)]
        if idx not in selected_indices:
            selected_indices.append(idx)

    # Results containers
    noise_correlations = []
    perturbation_jaccards = []
    attr_variances = []

    for sample_idx, idx in enumerate(selected_indices):
        x, y, evt = dataset[idx]
        x_np = x.numpy()

        # 🔧 FIXED: TimeSHAP-compliant baseline (mean across ALL features)
        baseline = np.tile(np.mean(x_np, axis=0, keepdims=True), (x_np.shape[0], 1))

        # 1. Baseline Attribution
        base_attr = temporal_attribution_enhanced(model, dataset, x_np, baseline, 4, device)

        # 2. Noise Robustness (5 runs with 5% noise)
        sample_correlations = []
        noisy_attributions = []

        for _ in range(5):
            noise = np.random.normal(0, 0.05, x_np.shape)
            noisy_x = x_np + noise

            # Recompute baseline for noisy input
            noisy_baseline = np.tile(np.mean(noisy_x, axis=0, keepdims=True), (noisy_x.shape[0], 1))
            noisy_attr = temporal_attribution_enhanced(model, dataset, noisy_x, noisy_baseline, 4, device)
            noisy_attributions.append(noisy_attr)

            corr, _ = spearmanr(base_attr, noisy_attr)
            sample_correlations.append(corr if not np.isnan(corr) else 0.0)

        noise_correlations.extend(sample_correlations)

        # 🔧 FIXED: Attribution Variance (variance of attribution values, not correlations)
        noisy_attributions = np.stack(noisy_attributions, axis=0)  # [K=5, T_segments]
        attr_var = np.mean(np.var(noisy_attributions, axis=0))  # Mean variance across time segments
        attr_variances.append(attr_var)

        # 3. 🔧 FIXED: Renamed to "Perturbation Consistency"
        top_k = min(3, len(base_attr))
        base_top = set(np.argsort(base_attr)[-top_k:])
        sample_jaccards = []

        for seed in [1, 2, 3, 4, 5]:
            seed_everything(seed)

            # Micro perturbation to test consistency
            micro_noise = np.random.normal(0, 0.01, x_np.shape)
            pert_x = x_np + micro_noise
            pert_baseline = np.tile(np.mean(pert_x, axis=0, keepdims=True), (pert_x.shape[0], 1))
            pert_attr = temporal_attribution_enhanced(model, dataset, pert_x, pert_baseline, 4, device)

            pert_top = set(np.argsort(pert_attr)[-top_k:])
            intersection = len(base_top.intersection(pert_top))
            union = len(base_top.union(pert_top))
            jaccard = intersection / union if union > 0 else 0.0
            sample_jaccards.append(jaccard)

        perturbation_jaccards.extend(sample_jaccards)

    # Compute summary statistics
    mean_noise_corr = np.mean(noise_correlations)
    mean_jaccard = np.mean(perturbation_jaccards)
    mean_attr_var = np.mean(attr_variances)

    print(f"   ✅ Noise Robustness (Spearman): {mean_noise_corr:.4f}")
    print(f"   ✅ Perturbation Consistency (Jaccard): {mean_jaccard:.4f}")
    print(f"   ✅ Attribution Variance: {mean_attr_var:.6f}")

    STABILITY_LOG.append({
        "Model": model_name,
        "Noise_Robustness_Spearman": mean_noise_corr,
        "Perturbation_Consistency_Jaccard": mean_jaccard,
        "Attribution_Variance": mean_attr_var,
        "Output_Sensitivity": mean_attr_var,  # Same as Attribution Variance
        "N_Samples": len(selected_indices)
    })

    return {
        "noise_correlations": noise_correlations,
        "perturbation_jaccards": perturbation_jaccards,
        "attr_variances": attr_variances
    }
    

# ==================================================================================
# 10. 🔥 CRITICAL PLOTS FOR THESIS
# ==================================================================================

def plot_event_vs_normal_error(results_df, save_path="event_vs_normal_error.png"):
    """
    📊 Plot A: Event-Focused Error vs Normal Error Comparison
    Purpose: Prove that the model reduces errors on sudden fluctuations
    while maintaining overall accuracy.
    """
    print("\n🔥 Creating Event vs Normal Error plot...")

    if 'Model' not in results_df.columns or 'MAE' not in results_df.columns:
        print("   ⚠️ Missing required columns")
        return

    models = results_df['Model'].values
    mae_total = results_df['MAE'].values
    sf_mae = results_df['SF_MAE'].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Event-Focused Error Analysis", fontsize=16, fontweight='bold')

    x = np.arange(len(models))
    width = 0.35

    # LEFT: Side-by-side comparison
    bars1 = axes[0].bar(x - width/2, mae_total, width, label='Overall MAE',
                        color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, sf_mae, width, label='Event MAE (SF_MAE)',
                        color='orangered', edgecolor='black', linewidth=1.5)

    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].set_title('Overall vs Event Error', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=11)
    axes[0].legend(fontsize=11, loc='upper left')
    axes[0].grid(alpha=0.3, axis='y', linestyle='--')

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    '{:.4f}'.format(height), ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    '{:.4f}'.format(height), ha='center', va='bottom', fontsize=9, color='red')

    # RIGHT: Ratio
    error_ratio = sf_mae / (mae_total + 1e-6)
    colors_ratio = ['green' if r < 1.5 else 'orange' if r < 2.0 else 'red' for r in error_ratio]

    bars3 = axes[1].bar(x, error_ratio, color=colors_ratio,
                       edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1].axhline(1.0, color='black', linestyle='--', linewidth=2, label='Equal Error (1.0)')

    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Error Ratio (Event / Overall)', fontsize=12)
    axes[1].set_title('Event Error Magnification', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=11)
    axes[1].legend(fontsize=10, loc='upper left')
    axes[1].grid(alpha=0.3, axis='y', linestyle='--')

    for i, (bar, ratio) in enumerate(zip(bars3, error_ratio)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    '{:.2f}x'.format(ratio), ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Saved: " + save_path)

    best_idx = np.argmin(error_ratio)
    print("   📊 Best model: {} (ratio: {:.2f}x)".format(models[best_idx], error_ratio[best_idx]))

def plot_event_distribution(df, threshold, save_path="event_distribution.png"):
    """
    📊 Plot B: Event Distribution & Threshold Validation
    Purpose: Show that events are rare but critical, validate threshold choice.
    """
    print("\n📊 Creating Event Distribution plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Event Detection & Distribution Analysis", fontsize=16, fontweight='bold')

    # TOP: Histogram
    abs_delta = df['abs_delta'].values
    n_events = (abs_delta > threshold).sum()
    event_pct = n_events / len(abs_delta) * 100

    axes[0].hist(abs_delta, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Non-events')
    axes[0].hist(abs_delta[abs_delta > threshold], bins=20, color='red', edgecolor='black', 
                alpha=0.8, label=f'Events (>{threshold:.4f})')

    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'95th Percentile Threshold')

    axes[0].set_xlabel('|Δ log(EC)|', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Log-Differences', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, axis='y')

    # Add statistics box
    stats_text = f'Total samples: {len(abs_delta)}\nEvents: {n_events} ({event_pct:.2f}%)\nThreshold: {threshold:.4f}'
    axes[0].text(0.98, 0.95, stats_text, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # BOTTOM: Time series with event regions
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        axes[1].plot(dates, df['OT_log'], color='blue', linewidth=0.8, label='Log(EC)')

        # Shade event regions
        event_mask = abs_delta > threshold
        axes[1].fill_between(dates, df['OT_log'].min(), df['OT_log'].max(),
                            where=event_mask, color='red', alpha=0.2, label='Event regions')

        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Log(EC)', fontsize=12)
        axes[1].set_title('Time Series with Event Regions Highlighted', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Saved: " + save_path)
    print(f"   📊 Events: {n_events}/{len(abs_delta)} ({event_pct:.2f}%)")

def plot_imf_attribution(df, n_imfs=5, save_path="imf_attribution.png"):
    """
    📊 Plot C: IMF Variance Attribution Analysis
    Purpose: Show which IMFs contribute most to fluctuations (justify CEEMD).
    """
    print("\n🌊 Creating IMF Attribution plot...")

    imf_cols = [f'IMF_{i}' for i in range(n_imfs) if f'IMF_{i}' in df.columns]

    if len(imf_cols) == 0:
        print("   ⚠️ No IMF columns found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("CEEMD Decomposition: IMF Variance Attribution", fontsize=16, fontweight='bold')

    # TOP-LEFT: Variance contribution bar chart
    variances = [df[col].var() for col in imf_cols]
    if 'residue' in df.columns:
        variances.append(df['residue'].var())
        labels = imf_cols + ['Residue']
    else:
        labels = imf_cols

    total_var = sum(variances)
    percentages = [v/total_var*100 for v in variances]

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    bars = axes[0, 0].bar(range(len(labels)), percentages, color=colors, edgecolor='black', linewidth=1.5)

    axes[0, 0].set_xlabel('Component', fontsize=12)
    axes[0, 0].set_ylabel('Variance Contribution (%)', fontsize=12)
    axes[0, 0].set_title('Variance Attribution', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].grid(alpha=0.3, axis='y')

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # TOP-RIGHT: Cumulative energy
    cumulative = np.cumsum(percentages)
    axes[0, 1].plot(range(len(labels)), cumulative, marker='o', linewidth=2, 
                   markersize=8, color='darkblue')
    axes[0, 1].axhline(90, color='red', linestyle='--', label='90% threshold')
    axes[0, 1].fill_between(range(len(labels)), 0, cumulative, alpha=0.2)

    axes[0, 1].set_xlabel('Component', fontsize=12)
    axes[0, 1].set_ylabel('Cumulative Energy (%)', fontsize=12)
    axes[0, 1].set_title('Cumulative Variance', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(range(len(labels)))
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # BOTTOM: IMF_0 (high-freq) vs Residue (low-freq) time series
    if 'date' in df.columns and len(imf_cols) > 0:
        dates = pd.to_datetime(df['date'])

        # Show first 500 samples for clarity
        n_show = min(500, len(dates))

        axes[1, 0].plot(dates[:n_show], df[imf_cols[0]][:n_show], 
                       color='red', linewidth=0.8, label=f'{imf_cols[0]} (High-freq)')
        axes[1, 0].set_xlabel('Date', fontsize=12)
        axes[1, 0].set_ylabel('Amplitude', fontsize=12)
        axes[1, 0].set_title(f'{imf_cols[0]}: High-Frequency Fluctuations', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        if 'residue' in df.columns:
            axes[1, 1].plot(dates[:n_show], df['residue'][:n_show],
                           color='blue', linewidth=1.2, label='Residue (Low-freq trend)')
            axes[1, 1].set_xlabel('Date', fontsize=12)
            axes[1, 1].set_ylabel('Amplitude', fontsize=12)
            axes[1, 1].set_title('Residue: Low-Frequency Trend', fontsize=13, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ Saved: " + save_path)
    print(f"   📊 Top contributor: {labels[np.argmax(percentages)]} ({max(percentages):.1f}%)")

# ==================================================================================
# 11. 🔧 FIXED: plot_runtime_and_stability()
# ==================================================================================

def plot_runtime_and_stability(save_runtime="runtime_comparison.png",
                               save_stability="stability_analysis.png"):
    """
    🔧 FIXED: Check if columns exist before groupby
    📊 Runtime & Stability Visualization
    """
    print(f"\n📊 Creating Runtime & Stability plots...")

    # Check if logs are empty
    if len(RUNTIME_LOG) == 0:
        print("   ⚠️ RUNTIME_LOG is empty, skipping runtime plot")
    else:
        df_runtime = pd.DataFrame(RUNTIME_LOG)
        print(f"   📊 Runtime log columns: {df_runtime.columns.tolist()}")

        # Create runtime plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Runtime Comparison", fontsize=16, fontweight='bold')

        # Check if 'Model' column exists
        if 'Model' in df_runtime.columns:
            models = df_runtime['Model'].unique()

            # LEFT: Training time
            if 'Training_Time' in df_runtime.columns:
                train_times = []
                for model in models:
                    val = df_runtime[df_runtime['Model'] == model]['Training_Time'].values
                    train_times.append(val[0] if len(val) > 0 else 0)

                bars1 = axes[0].bar(range(len(models)), train_times, 
                                   color='steelblue', edgecolor='black', linewidth=1.5)
                axes[0].set_xlabel('Model', fontsize=12)
                axes[0].set_ylabel('Training Time (seconds)', fontsize=12)
                axes[0].set_title('Training Time', fontsize=13, fontweight='bold')
                axes[0].set_xticks(range(len(models)))
                axes[0].set_xticklabels(models, fontsize=11)
                axes[0].grid(alpha=0.3, axis='y')

                for bar in bars1:
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

            # RIGHT: Inference time
            if 'Inference_Time' in df_runtime.columns:
                infer_times = []
                for model in models:
                    val = df_runtime[df_runtime['Model'] == model]['Inference_Time'].values
                    infer_times.append(val[0] if len(val) > 0 else 0)

                bars2 = axes[1].bar(range(len(models)), infer_times, 
                                   color='coral', edgecolor='black', linewidth=1.5)
                axes[1].set_xlabel('Model', fontsize=12)
                axes[1].set_ylabel('Inference Time (ms/sample)', fontsize=12)
                axes[1].set_title('Inference Time', fontsize=13, fontweight='bold')
                axes[1].set_xticks(range(len(models)))
                axes[1].set_xticklabels(models, fontsize=11)
                axes[1].grid(alpha=0.3, axis='y')

                for bar in bars2:
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_runtime, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {save_runtime}")

    # Check if stability log is empty
    if len(STABILITY_LOG) == 0:
        print("   ⚠️ STABILITY_LOG is empty, skipping stability plot")
    else:
        df_stability = pd.DataFrame(STABILITY_LOG)
        print(f"   📊 Stability log columns: {df_stability.columns.tolist()}")

        # Create stability plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Stability Analysis", fontsize=16, fontweight='bold')

        if 'Model' in df_stability.columns:
            models = df_stability['Model'].unique()

            # LEFT: Noise Robustness
            if 'Noise_Robustness_Spearman' in df_stability.columns:
                values = []
                for model in models:
                    val = df_stability[df_stability['Model'] == model]['Noise_Robustness_Spearman'].values
                    values.append(val[0] if len(val) > 0 else 0)

                bars1 = axes[0].bar(range(len(models)), values, 
                                   color='skyblue', edgecolor='black', linewidth=1.5)
                axes[0].set_xlabel('Model', fontsize=12)
                axes[0].set_ylabel('Spearman Correlation', fontsize=12)
                axes[0].set_title('Noise Robustness', fontsize=13, fontweight='bold')
                axes[0].set_xticks(range(len(models)))
                axes[0].set_xticklabels(models, fontsize=11)
                axes[0].set_ylim([0, 1.05])
                axes[0].grid(alpha=0.3, axis='y')

                for bar in bars1:
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

            # MIDDLE: Perturbation Consistency
            if 'Perturbation_Consistency_Jaccard' in df_stability.columns:
                values = []
                for model in models:
                    val = df_stability[df_stability['Model'] == model]['Perturbation_Consistency_Jaccard'].values
                    values.append(val[0] if len(val) > 0 else 0)

                bars2 = axes[1].bar(range(len(models)), values, 
                                   color='lightgreen', edgecolor='black', linewidth=1.5)
                axes[1].set_xlabel('Model', fontsize=12)
                axes[1].set_ylabel('Jaccard Index', fontsize=12)
                axes[1].set_title('Perturbation Consistency', fontsize=13, fontweight='bold')
                axes[1].set_xticks(range(len(models)))
                axes[1].set_xticklabels(models, fontsize=11)
                axes[1].set_ylim([0, 1.05])
                axes[1].grid(alpha=0.3, axis='y')

                for bar in bars2:
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

            # RIGHT: Attribution Variance
            if 'Attribution_Variance' in df_stability.columns:
                values = []
                for model in models:
                    val = df_stability[df_stability['Model'] == model]['Attribution_Variance'].values
                    values.append(val[0] if len(val) > 0 else 0)

                bars3 = axes[2].bar(range(len(models)), values, 
                                   color='salmon', edgecolor='black', linewidth=1.5)
                axes[2].set_xlabel('Model', fontsize=12)
                axes[2].set_ylabel('Variance', fontsize=12)
                axes[2].set_title('Attribution Variance', fontsize=13, fontweight='bold')
                axes[2].set_xticks(range(len(models)))
                axes[2].set_xticklabels(models, fontsize=11)
                axes[2].grid(alpha=0.3, axis='y')

                for bar in bars3:
                    height = bar.get_height()
                    axes[2].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_stability, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {save_stability}")

    print("   ✅ Runtime & Stability plots completed")

# ==================================================================================
# 12. MAIN v3.6 FINAL FIXED WITH ALL PLOTS
# ==================================================================================

def main():
    """Main v3.6 FINAL FIXED - Bug-free thesis version with ALL PLOTS"""
    print("="*80)
    print("🌊 WATER QUALITY FORECASTING v3.6 FINAL FIXED + FULL POWER")
    print("="*80)
    print("✅ ALL CRITICAL BUGS FIXED:")
    print("   🐛 Inference time (per-sample, not per-batch)")
    print("   🐛 Attribution variance (variance of attributions, not correlations)")
    print("   🐛 Baseline strategy (TimeSHAP-compliant)")
    print("   🐛 Metric names (Perturbation Consistency)")
    print("   🐛 Runtime aggregation (no duplicate bars)")
    print("   🔥 CEEMD Full Power (100 trials, all data, 5 IMFs)")
    print("="*80)

    # Seed for reproducibility
    seed_everything(42)

    # [1/8] LOAD DATA
    df_ec, df_ph = load_data_source_separate()
    if df_ec is None or df_ph is None:
        df_ec, df_ph = load_data_source_api()
    if df_ec is None or df_ph is None:
        print("❌ Cannot load data")
        return

    # [2/8] CEEMD DECOMPOSITION (FULL POWER)
    n_imfs_ec = 5
    try:
        df_ec, n_imfs_ec = apply_ceemd_decomposition(df_ec, n_imfs=5)
    except Exception as e:
        print(f"   ⚠️ CEEMD skipped: {e}")

    # [2.5/8] EVENT DETECTION & DISTRIBUTION PLOT
    print("\n" + "="*80)
    print("📊 [2.5/8] Event Distribution Analysis")
    print("="*80)
    
    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df_ec, CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size'], False
    )
    train_size = split_info['train_size']
    train_end_idx = train_size + CONFIG['seq_len']
    df_train = df_ec.iloc[:train_end_idx].copy()
    
    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )
    df_ec = detect_events_from_threshold(df_ec, event_threshold)
    
    # ✅ CRITICAL PLOT 1
    plot_event_distribution(df_ec, event_threshold, "event_distribution.png")
    
    # [2.6/8] IMF ATTRIBUTION PLOT
    if 'IMF_0' in df_ec.columns:
        print("\n🌊 [2.6/8] IMF Contribution Analysis")
        # ✅ CRITICAL PLOT 2
        plot_imf_attribution(df_ec, n_imfs=n_imfs_ec, save_path="imf_attribution.png")


    # [3/8] STANDARD MODEL COMPARISON
    print("\n" + "="*80)
    print("📊 [3/8] STANDARD: Model Comparison")
    print("="*80)

    results = []
    for model_type in ["NLinear", "DLinear", "CALinear"]:
        metrics, model, dataset, test_loader, df_updated = run_experiment(
            df_ec.copy(), "EC", "Files", model_type
        )
        results.append(metrics)

    final_df = pd.DataFrame(results)
    final_df.to_csv("comparison_results.csv", index=False)
    
    # ✅ CRITICAL PLOT 3
    plot_event_vs_normal_error(final_df, "event_vs_normal_error.png")

    # [4/8] MULTI-HORIZON
    horizons = [12, 24, 48]
    horizon_results = run_multi_horizon_experiments(
        df_ec.copy(), "EC", "Files", "CALinear", horizons
    )
    horizon_results.to_csv("horizon_comparison.csv", index=False)

    # [5/8] ABLATION
    ablation_df = run_ablation_study(df_ec.copy(), "EC", "Files")
    ablation_df.to_csv("ablation_study.csv", index=False)

    # [6/8] ALPHA
    alpha_df = run_alpha_sensitivity_analysis(df_ec.copy(), "EC", "Files", [1.0, 3.0, 5.0])
    alpha_df.to_csv("alpha_sensitivity.csv", index=False)

    # [7/8] STABILITY
    for model_type in ["NLinear", "DLinear", "CALinear"]:
        metrics, model, dataset, test_loader, _ = run_experiment(
            df_ec.copy(), "EC", "Files", model_type
        )
        run_stability_analysis(model, dataset, test_loader, device, model_type)

    # [8/8] SAVE REPORTS
    df_runtime = pd.DataFrame(RUNTIME_LOG)
    df_stability = pd.DataFrame(STABILITY_LOG)
    df_runtime.to_csv("runtime_report.csv", index=False)
    df_stability.to_csv("stability_report.csv", index=False)

    plot_runtime_and_stability()

    print("\n✅ v3.6 FINAL FIXED COMPLETE!")

if __name__ == "__main__":
    main()
