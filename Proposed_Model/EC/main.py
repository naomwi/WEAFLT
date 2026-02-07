"""
Main Entry Point for Proposed Model: CA-CEEMDAN-LTSF
Target: EC (Electrical Conductivity)
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, DATA_CONFIG, TRAIN_CONFIG, FEATURE_CONFIG, LOSS_CONFIG,
    DECOMPOSITION_CONFIG, HORIZONS, EXPERIMENTS, RESULTS_DIR, MODEL_DIR,
    CEEMDAN_CACHE_DIR, CACHE_DIR, get_config_summary
)
from train import train_all_imf_models, train_all_imf_models_parallel
from models import DLinear, NLinear, IMFAggregator
from utils.decomposition import get_or_create_imfs
from utils.data_loader import load_raw_data, create_dataloaders
from utils.feature_engineering import create_change_aware_features, create_event_flags
from utils.losses import EventWeightedLoss
from utils.metrics import calculate_all_metrics, print_metrics
from utils.plotting import plot_prediction


def run_experiment(
    model_type: str,
    horizon: int,
    data: np.ndarray,
    device: torch.device,
    verbose: bool = True
) -> dict:
    """
    Run experiment for a specific model and horizon.

    Pipeline:
    1. Compute change-aware features from ORIGINAL data
    2. Compute event flags from ORIGINAL data
    3. CEEMDAN decomposition
    4. Train per-IMF models
    5. Aggregate predictions
    6. Evaluate
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {model_type.upper()} | Horizon: {horizon}")
        print(f"{'='*60}")

    seq_len = DATA_CONFIG['seq_len']
    n_imfs = DECOMPOSITION_CONFIG['max_imfs']

    # =========================================================================
    # Step 1-2: Features and Event flags from ORIGINAL data
    # =========================================================================
    if verbose:
        print("\nStep 1-2: Computing features from ORIGINAL data...")

    train_end_idx = int(len(data) * DATA_CONFIG['train_ratio'])

    # Change-aware features
    feat_result = create_change_aware_features(
        data,
        rolling_std_window=FEATURE_CONFIG['rolling_std_window'],
        rolling_zscore_window=FEATURE_CONFIG['rolling_zscore_window']
    )
    features = feat_result['features']

    # Event flags (threshold from training data only)
    event_flags, threshold = create_event_flags(
        data,
        percentile=FEATURE_CONFIG['event_percentile'],
        train_end_idx=train_end_idx
    )

    if verbose:
        print(f"  Features: {feat_result['feature_names']}")
        print(f"  Events: {event_flags.sum():.0f} (threshold={threshold:.4f})")

    # =========================================================================
    # Step 3: CEEMDAN Decomposition
    # =========================================================================
    if verbose:
        print("\nStep 3: CEEMDAN Decomposition...")

    # Try to load from CEEMDAN_models cache first
    try:
        from utils.decomposition import load_cached_imfs
        result = load_cached_imfs(CEEMDAN_CACHE_DIR, prefix="ec", n_imfs=n_imfs)
        imfs = result['imfs']
        residue = result['residue']
        if verbose:
            print(f"  Loaded {len(imfs)} IMFs from CEEMDAN_models cache")
    except FileNotFoundError:
        result = get_or_create_imfs(
            data, CACHE_DIR, prefix="ec",
            n_imfs=n_imfs,
            trials=DECOMPOSITION_CONFIG['trials'],
            epsilon=DECOMPOSITION_CONFIG['epsilon']
        )
        imfs = result['imfs']
        residue = result['residue']

    # =========================================================================
    # Step 4: Train per-IMF models
    # =========================================================================
    if verbose:
        print(f"\nStep 4: Training {n_imfs + 1} models...")

    train_config = {
        'epochs': TRAIN_CONFIG['epochs'],
        'learning_rate': TRAIN_CONFIG['learning_rate'],
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'early_stopping_patience': TRAIN_CONFIG['early_stopping_patience'],
        'event_weight': LOSS_CONFIG['event_weight'],
        'batch_size': DATA_CONFIG['batch_size'],
    }

    save_dir = MODEL_DIR / model_type

    # Use parallel training for faster GPU utilization
    models, scalers = train_all_imf_models_parallel(
        imfs, residue, features, event_flags,
        model_type=model_type,
        seq_len=seq_len,
        pred_len=horizon,
        device=device,
        config=train_config,
        save_dir=save_dir,
        max_workers=4,  # Train 4 IMF models in parallel
        verbose=verbose
    )

    # =========================================================================
    # Step 5: Evaluate on test set
    # =========================================================================
    if verbose:
        print("\nStep 5: Evaluating on test set...")

    # Get predictions from all models
    all_preds = []
    all_actuals = []

    components = list(imfs) + [residue]

    for i, (component, model, (imf_scaler, feat_scaler)) in enumerate(zip(components, models, scalers)):
        # Create test loader
        _, _, test_loader, _, _ = create_dataloaders(
            component, features, event_flags,
            seq_len=seq_len, pred_len=horizon,
            batch_size=DATA_CONFIG['batch_size']
        )

        # Get predictions
        model.eval()
        preds = []
        actuals = []

        with torch.no_grad():
            for batch in test_loader:
                x, y, _ = batch
                x = x.to(device)
                pred = model(x)
                preds.append(pred.cpu().numpy())
                actuals.append(y.numpy())

        preds = np.concatenate(preds, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        # Inverse transform
        preds_inv = imf_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        actuals_inv = imf_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        # Reshape back
        n_samples = len(preds)
        preds_inv = preds_inv.reshape(n_samples, horizon)
        actuals_inv = actuals_inv.reshape(n_samples, horizon)

        all_preds.append(preds_inv)
        all_actuals.append(actuals_inv)

    # =========================================================================
    # Step 6: Aggregate and compute final metrics
    # =========================================================================
    # Stack predictions: (n_components, n_samples, horizon)
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Aggregation: Simple sum (IMFs + Residue should reconstruct original signal)
    # Note: For attention-based aggregation, train the aggregator separately
    final_pred = all_preds.sum(axis=0)
    final_actual = all_actuals.sum(axis=0)

    # Log IMF contribution (energy-based)
    if verbose:
        n_components = len(components)
        energies = [np.var(all_preds[i]) for i in range(n_components)]
        total_energy = sum(energies)
        print(f"\n  IMF Energy Contribution:")
        for i, e in enumerate(energies):
            name = f"IMF_{i+1}" if i < 12 else "Residue"
            contribution = e / total_energy * 100 if total_energy > 0 else 0
            print(f"    {name}: {contribution:.2f}%")

    # Use last point for metrics
    pred_last = final_pred[:, -1]
    actual_last = final_actual[:, -1]

    metrics = calculate_all_metrics(actual_last, pred_last)

    if verbose:
        print_metrics(metrics, f"{model_type.upper()} | Horizon {horizon}")

    # Save results
    save_results(pred_last, actual_last, metrics, model_type, horizon)

    return {
        'model': model_type,
        'horizon': horizon,
        'metrics': metrics,
        'predictions': pred_last,
        'actuals': actual_last
    }


def save_results(predictions, actuals, metrics, model_type, horizon):
    """Save results to files."""
    metrics_dir = RESULTS_DIR / "metrics"
    series_dir = RESULTS_DIR / "series"
    plots_dir = RESULTS_DIR / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    metrics_df = pd.DataFrame([{
        'Model': model_type,
        'Horizon': horizon,
        **metrics
    }])
    metrics_df.to_csv(metrics_dir / f"{model_type}_h{horizon}.csv", index=False)

    # Series
    series_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions
    })
    series_df.to_csv(series_dir / f"series_{model_type}_P{horizon}_EC.csv", index=False)

    # Plot comparison
    plot_path = plots_dir / f"series_{model_type}_P{horizon}_EC.png"
    plot_prediction(
        actual=actuals,
        predicted=predictions,
        title=f"Comparison: series_{model_type}_P{horizon}_EC",
        save_path=plot_path
    )


def main():
    parser = argparse.ArgumentParser(description='CA-CEEMDAN-LTSF for EC Prediction')
    parser.add_argument('--model', '-m', type=str, default='dlinear',
                        choices=['dlinear', 'nlinear'], help='Model type')
    parser.add_argument('--horizon', '-H', type=int, default=None,
                        help='Prediction horizon (default: all)')
    parser.add_argument('--all', '-a', action='store_true', help='Run all experiments')
    parser.add_argument('--device', '-d', type=str, default=None, help='Device')

    args = parser.parse_args()

    # Config summary
    get_config_summary()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading data...")
    data_path = DATA_DIR / DATA_CONFIG['data_file']
    df, data = load_raw_data(str(data_path), DATA_CONFIG['target_col'])
    print(f"  Loaded {len(data)} samples")

    # Determine experiments
    horizons = [args.horizon] if args.horizon else HORIZONS
    models = list(EXPERIMENTS.keys()) if args.all else [args.model]

    # Run experiments
    all_results = []
    start_time = time.time()

    for model_type in models:
        for horizon in horizons:
            try:
                result = run_experiment(model_type, horizon, data, device)
                all_results.append(result)
            except Exception as e:
                print(f"\nERROR in {model_type} h{horizon}: {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED IN {elapsed/60:.1f} MINUTES")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
