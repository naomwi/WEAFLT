"""
Main Entry for CEEMD Baselines
CEEMD + DLinear/NLinear + Standard + MSE
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DATA_CONFIG, TRAIN_CONFIG, DECOMPOSITION_CONFIG, HORIZONS, RESULTS_DIR, MODEL_DIR, CACHE_DIR
from models import DLinear, NLinear
from utils import get_or_create_imfs, create_dataloaders, calculate_all_metrics, print_metrics, plot_prediction
from utils.data_loader import load_raw_data


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, learning_rate=None, patience=10, early_stopping_patience=None, **kwargs):
    # Handle both 'lr' and 'learning_rate' parameter names
    lr = learning_rate if learning_rate is not None else lr
    patience = early_stopping_patience if early_stopping_patience is not None else patience

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val = float('inf')
    patience_cnt = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def train_single_imf(idx, comp, model_type, seq_len, horizon, device, train_config, print_lock=None, verbose=True):
    """
    Train a single IMF model.

    Note: No scaling is applied to IMFs. CEEMD preserves the mathematical property:
    original_signal = sum(IMFs) + Residue

    Predictions can be summed directly without inverse_transform.
    """
    train_ld, val_ld, test_ld, _ = create_dataloaders(comp, seq_len, horizon)

    if model_type == 'dlinear':
        model = DLinear(seq_len, horizon)
    else:
        model = NLinear(seq_len, horizon)

    model, val_loss = train_model(model, train_ld, val_ld, device, **train_config)

    # Evaluate
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_ld:
            preds.append(model(x.to(device)).cpu().numpy())
            actuals.append(y.numpy())

    # NO inverse_transform - predictions are already in original IMF scale
    preds = np.concatenate(preds).flatten()
    actuals = np.concatenate(actuals).flatten()

    n_samples = len(preds) // horizon
    preds_reshaped = preds.reshape(n_samples, horizon)
    actuals_reshaped = actuals.reshape(n_samples, horizon)

    if verbose and print_lock:
        with print_lock:
            name = f"IMF_{idx+1}" if idx < 12 else "Residue"
            print(f"  {name}: val_loss={val_loss:.6f}")

    return idx, preds_reshaped, actuals_reshaped


def run_experiment(model_type, horizon, data, device, verbose=True):
    if verbose:
        print(f"\n{'='*50}")
        print(f"CEEMD + {model_type.upper()} | Horizon {horizon}")
        print(f"{'='*50}")

    seq_len = DATA_CONFIG['seq_len']
    n_imfs = DECOMPOSITION_CONFIG['max_imfs']

    # CEEMD decomposition
    result = get_or_create_imfs(
        data, CACHE_DIR, prefix="ec", n_imfs=n_imfs,
        trials=DECOMPOSITION_CONFIG['trials'],
        noise_width=DECOMPOSITION_CONFIG['noise_width']
    )
    imfs, residue = result['imfs'], result['residue']

    # Train per-IMF models SEQUENTIALLY (ThreadPoolExecutor has issues with CUDA)
    components = list(imfs) + [residue]
    all_preds = []
    all_actuals = []

    if verbose:
        print(f"\nTraining {len(components)} IMF models...")

    start_time = time.time()

    for i, comp in enumerate(components):
        name = f"IMF_{i+1}" if i < len(imfs) else "Residue"
        idx, preds, actuals = train_single_imf(
            i, comp, model_type, seq_len, horizon, device, TRAIN_CONFIG, None, verbose
        )
        all_preds.append(preds)
        all_actuals.append(actuals)

    elapsed = time.time() - start_time
    if verbose:
        print(f"  All models trained in {elapsed:.1f}s")

    # Sum all components
    final_pred = np.array(all_preds).sum(axis=0)[:, -1]
    final_actual = np.array(all_actuals).sum(axis=0)[:, -1]

    metrics = calculate_all_metrics(final_actual, final_pred)
    if verbose:
        print_metrics(metrics, f"  CEEMD-{model_type.upper()} H{horizon} Results:")

    # Save metrics
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'Model': model_type, 'Horizon': horizon, **metrics}]).to_csv(
        RESULTS_DIR / "metrics" / f"{model_type}_h{horizon}.csv", index=False
    )

    # Save series
    (RESULTS_DIR / "series").mkdir(parents=True, exist_ok=True)
    series_df = pd.DataFrame({'Actual': final_actual, 'Predicted': final_pred})
    series_path = RESULTS_DIR / "series" / f"series_{model_type}_P{horizon}_EC.csv"
    series_df.to_csv(series_path, index=False)

    # Plot comparison
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_DIR / "plots" / f"series_{model_type}_P{horizon}_EC.png"
    plot_prediction(
        actual=final_actual,
        predicted=final_pred,
        title=f"Comparison: series_{model_type}_P{horizon}_EC",
        save_path=plot_path
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='dlinear', choices=['dlinear', 'nlinear'])
    parser.add_argument('--horizon', '-H', type=int, default=None)
    parser.add_argument('--all', '-a', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df, data = load_raw_data(str(DATA_DIR / DATA_CONFIG['data_file']), DATA_CONFIG['target_col'])
    print(f"Loaded {len(data)} samples")

    horizons = HORIZONS if args.horizon is None else [args.horizon]
    models = ['dlinear', 'nlinear'] if args.all else [args.model]

    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    for m in models:
        for h in horizons:
            run_experiment(m, h, data, device)


if __name__ == "__main__":
    main()
