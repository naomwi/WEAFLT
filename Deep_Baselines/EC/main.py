"""
Main Entry for Deep Baselines
- LSTM: Uses CEEMDAN decomposition (per-IMF parallel)
- PatchTST, Transformer: End-to-end (NO decomposition)
Target: EC
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, HORIZONS, RESULTS_DIR
from models import LSTMModel, TransformerModel, PatchTST
from utils import create_dataloaders, create_imf_dataloaders, calculate_all_metrics, print_metrics, load_raw_data, plot_prediction

# For LSTM with CEEMDAN
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def get_model(model_type, seq_len, pred_len, config):
    if model_type == 'lstm':
        return LSTMModel(seq_len, pred_len, **config)
    elif model_type == 'transformer':
        return TransformerModel(seq_len, pred_len, **config)
    elif model_type == 'patchtst':
        return PatchTST(seq_len, pred_len, **config)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, patience=10, **kwargs):
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


def run_lstm_ceemdan(horizon, data, device, verbose=True):
    """
    Run LSTM with CEEMDAN decomposition (per-IMF parallel).
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"LSTM (CEEMDAN) | Horizon {horizon}")
        print(f"{'='*50}")

    # Import CEEMDAN utilities
    try:
        from Proposed_Model.EC.utils.decomposition import get_or_create_imfs
        CACHE_DIR = PROJECT_DIR / "Deep_Baselines" / "EC" / "cache"
    except ImportError:
        print("  Warning: Could not import CEEMDAN utilities, using raw signal")
        return run_experiment_endtoend('lstm', horizon, data, device, verbose)

    seq_len = DATA_CONFIG['seq_len']
    config = MODEL_CONFIG['lstm']

    # CEEMDAN decomposition
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = get_or_create_imfs(data, CACHE_DIR, prefix="ec", n_imfs=12)
    imfs, residue = result['imfs'], result['residue']

    if verbose:
        print(f"  CEEMDAN: {len(imfs)} IMFs + residue")

    # Train per-IMF LSTM models
    # Note: IMFs are NOT scaled. CEEMDAN preserves:
    # original_signal = sum(IMFs) + Residue (exact mathematical property)
    components = list(imfs) + [residue]
    all_preds, all_actuals = [], []

    start = time.time()
    for i, comp in enumerate(components):
        # Use IMF-specific dataloaders (no scaling)
        train_ld, val_ld, test_ld, _ = create_imf_dataloaders(comp, seq_len, horizon)

        model = LSTMModel(seq_len, horizon, **config)
        model, _ = train_model(model, train_ld, val_ld, device, **TRAIN_CONFIG)

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
        all_preds.append(preds.reshape(n_samples, horizon))
        all_actuals.append(actuals.reshape(n_samples, horizon))

        if verbose and (i == 0 or i == len(components) - 1):
            print(f"  IMF {i+1}/{len(components)} trained")

    if verbose:
        print(f"  Training done in {time.time()-start:.1f}s")

    # Sum all components - this now correctly reconstructs the original signal prediction
    final_pred = np.array(all_preds).sum(axis=0)[:, -1]
    final_actual = np.array(all_actuals).sum(axis=0)[:, -1]

    metrics = calculate_all_metrics(final_actual, final_pred)
    if verbose:
        print_metrics(metrics, f"  LSTM-CEEMDAN H{horizon} Results:")

    # Save results
    _save_results('lstm', horizon, final_actual, final_pred, metrics)

    return metrics


def run_experiment_endtoend(model_type, horizon, data, device, verbose=True):
    """
    Run end-to-end model (no decomposition).
    For: PatchTST, Transformer
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"{model_type.upper()} (End-to-End) | Horizon {horizon}")
        print(f"{'='*50}")

    seq_len = DATA_CONFIG['seq_len']
    config = MODEL_CONFIG[model_type]

    # Create dataloaders (from raw signal, no decomposition)
    train_ld, val_ld, test_ld, scaler = create_dataloaders(data, seq_len, horizon)

    # Create model
    model = get_model(model_type, seq_len, horizon, config)
    if verbose:
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    start = time.time()
    model, best_val = train_model(model, train_ld, val_ld, device, **TRAIN_CONFIG)
    if verbose:
        print(f"  Training done in {time.time()-start:.1f}s, best_val={best_val:.6f}")

    # Evaluate
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_ld:
            preds.append(model(x.to(device)).cpu().numpy())
            actuals.append(y.numpy())

    preds = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.concatenate(actuals).reshape(-1, 1)).flatten()

    # Use last point
    n_samples = len(preds) // horizon
    preds = preds.reshape(n_samples, horizon)[:, -1]
    actuals = actuals.reshape(n_samples, horizon)[:, -1]

    metrics = calculate_all_metrics(actuals, preds)
    if verbose:
        print_metrics(metrics, f"  {model_type.upper()} H{horizon} Results:")

    # Save results
    _save_results(model_type, horizon, actuals, preds, metrics)

    return metrics


def _save_results(model_type, horizon, actuals, preds, metrics):
    """Save metrics, series, and plot."""
    # Save metrics
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'Model': model_type, 'Horizon': horizon, **metrics}]).to_csv(
        RESULTS_DIR / "metrics" / f"{model_type}_h{horizon}.csv", index=False
    )

    # Save series
    (RESULTS_DIR / "series").mkdir(parents=True, exist_ok=True)
    series_df = pd.DataFrame({'Actual': actuals, 'Predicted': preds})
    series_path = RESULTS_DIR / "series" / f"series_{model_type}_P{horizon}_EC.csv"
    series_df.to_csv(series_path, index=False)

    # Plot comparison
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_DIR / "plots" / f"series_{model_type}_P{horizon}_EC.png"
    plot_prediction(
        actual=actuals,
        predicted=preds,
        title=f"Comparison: series_{model_type}_P{horizon}_EC",
        save_path=plot_path
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='lstm', choices=['lstm', 'patchtst', 'transformer'])
    parser.add_argument('--horizon', '-H', type=int, default=None)
    parser.add_argument('--all', '-a', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df, data = load_raw_data(str(DATA_DIR / DATA_CONFIG['data_file']), DATA_CONFIG['target_col'])
    print(f"Loaded {len(data)} samples")

    horizons = HORIZONS if args.horizon is None else [args.horizon]
    models = ['lstm', 'patchtst', 'transformer'] if args.all else [args.model]

    for m in models:
        for h in horizons:
            if m == 'lstm':
                # LSTM uses CEEMDAN
                run_lstm_ceemdan(h, data, device)
            else:
                # PatchTST, Transformer are end-to-end
                run_experiment_endtoend(m, h, data, device)


if __name__ == "__main__":
    main()
