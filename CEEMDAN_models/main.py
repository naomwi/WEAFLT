"""
Main Entry Point for CEEMDAN-based Forecasting
Simple, clean implementation for water quality (EC) prediction

Usage:
    python main.py                    # Run full pipeline
    python main.py --decompose-only   # Only run CEEMDAN decomposition
    python main.py --train-only       # Only train models (requires decomposed data)
    python main.py --evaluate-only    # Only evaluate (requires trained models)
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from config import (
    DATA_CONFIG, CEEMDAN_CONFIG, TRAIN_CONFIG, MODEL_CONFIG,
    DATA_DIR, IMF_DIR, MODEL_DIR, RESULTS_DIR,
    HORIZONS, AVAILABLE_MODELS
)
from utils.data_loader import load_raw_data, get_target_data
from utils.ceemdan_decomposition import decompose_and_save, load_imfs
from utils.metrics import print_metrics
from train import train_all_components, get_device
from evaluate import run_evaluation, generate_comparison_report, plot_predictions
from plot_visual import plot_all


def setup_directories():
    """Create necessary directories."""
    dirs = [
        DATA_DIR,
        IMF_DIR / 'train',
        IMF_DIR / 'test',
        MODEL_DIR,
        RESULTS_DIR / 'plots',
        RESULTS_DIR / 'metrics',
        RESULTS_DIR / 'series',  # For Actual vs Predicted CSV files
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories created.")


def check_cached_imfs() -> bool:
    """Check if IMFs are already cached."""
    n_imfs = CEEMDAN_CONFIG['n_imfs']

    # Check if all IMF files exist
    for i in range(n_imfs):
        imf_file = IMF_DIR / f"ec_imf_{i+1}.npy"
        if not imf_file.exists():
            return False

    # Check residue
    residue_file = IMF_DIR / "ec_residue.npy"
    if not residue_file.exists():
        return False

    return True


def run_decomposition(target_data: np.ndarray, force: bool = False) -> dict:
    """
    Run CEEMDAN decomposition and save IMFs.
    Uses cached IMFs if available (same as ceemdan_EVloss).

    Args:
        target_data: Raw target (EC) data
        force: Force re-decomposition even if cached

    Returns:
        Dictionary with IMF data
    """
    print("\n" + "="*60)
    print("STEP 1: CEEMDAN DECOMPOSITION")
    print("="*60)

    # Check for cached IMFs (avoid slow re-computation)
    if not force and check_cached_imfs():
        print("Found cached IMFs! Loading from files...")
        imf_result = load_imfs(IMF_DIR, prefix='ec', n_imfs=CEEMDAN_CONFIG['n_imfs'])
        print(f"  Loaded {len(imf_result['imfs'])} IMFs + residue from cache")
        return imf_result

    print("No cached IMFs found. Running CEEMDAN decomposition...")

    # Decompose and save
    _, imf_result = decompose_and_save(
        pd.Series(target_data),
        IMF_DIR,
        prefix='ec'
    )

    print(f"\nDecomposition complete:")
    print(f"  - {len(imf_result['imfs'])} IMFs")
    print(f"  - 1 Residue")
    print(f"  - Files saved to: {IMF_DIR}")

    return imf_result


def run_training(imf_data: dict, device) -> dict:
    """
    Train all models on all components.

    Args:
        imf_data: Dictionary with 'imfs' and 'residue'
        device: Training device

    Returns:
        Dictionary with training results
    """
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)

    all_results = {}

    for model_name in AVAILABLE_MODELS:
        print(f"\n{'='*40}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*40}")

        model_results = {}

        for horizon in HORIZONS:
            print(f"\n--- Horizon: {horizon} hours ---")

            results = train_all_components(
                model_name=model_name,
                pred_len=horizon,
                device=device,
                imf_data=imf_data,
                save_models=True
            )

            model_results[horizon] = results

        all_results[model_name] = model_results

    print("\nTraining complete!")
    print(f"Models saved to: {MODEL_DIR}")

    return all_results


def run_full_evaluation(imf_data: dict, original_data: np.ndarray, device) -> pd.DataFrame:
    """
    Run evaluation for all models and horizons.

    Args:
        imf_data: IMF decomposition data
        original_data: Original target data
        device: Evaluation device

    Returns:
        DataFrame with all evaluation results
    """
    print("\n" + "="*60)
    print("STEP 3: EVALUATION")
    print("="*60)

    all_results = []

    for model_name in AVAILABLE_MODELS:
        print(f"\n{'='*40}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*40}")

        results_df = run_evaluation(
            model_name=model_name,
            horizons=HORIZONS,
            imf_data=imf_data,
            original_data=original_data,
            device=device
        )

        all_results.append(results_df)

    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)

    # Generate reports
    generate_comparison_report(final_results, RESULTS_DIR)

    # Save summary table
    summary_path = RESULTS_DIR / 'metrics' / 'summary_results.csv'
    final_results.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    return final_results


def print_summary(results_df: pd.DataFrame):
    """Print a nice summary of results."""
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Best model for each horizon
    print("\nBest Model by Horizon (lowest MAE):")
    print("-"*40)

    for horizon in sorted(results_df['Horizon'].unique()):
        horizon_data = results_df[results_df['Horizon'] == horizon]
        best_idx = horizon_data['MAE'].idxmin()
        best_row = results_df.loc[best_idx]
        print(f"  H={horizon:3d}h: {best_row['Model']:<12} MAE={best_row['MAE']:.4f}, R²={best_row['R2']:.4f}")

    # Overall best
    print("\n" + "-"*40)
    print("Overall Comparison (averaged across horizons):")
    print("-"*40)

    for model in AVAILABLE_MODELS:
        model_data = results_df[results_df['Model'] == model]
        avg_mae = model_data['MAE'].mean()
        avg_rmse = model_data['RMSE'].mean()
        avg_r2 = model_data['R2'].mean()
        print(f"  {model:<12}: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")


def main(args):
    """Main execution function."""
    print("="*60)
    print("CEEMDAN-BASED WATER QUALITY FORECASTING")
    print("Target: EC (Electrical Conductivity)")
    print("Models: LSTM, PatchTST, Transformer")
    print("="*60)

    # Setup
    setup_directories()
    device = get_device()

    # Load data
    print("\nLoading data...")
    try:
        df = load_raw_data()
        target_data = get_target_data(df, DATA_CONFIG['target_col'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the water quality data file is available.")
        return

    print(f"Data loaded: {len(target_data)} samples")
    print(f"EC stats: min={target_data.min():.2f}, max={target_data.max():.2f}, mean={target_data.mean():.2f}")

    # STEP 1: Decomposition
    if not args.train_only and not args.evaluate_only:
        imf_data = run_decomposition(target_data, force=args.force_decompose)
    else:
        # Load existing decomposition
        print("\nLoading existing decomposition...")
        imf_data = load_imfs(IMF_DIR, prefix='ec', n_imfs=CEEMDAN_CONFIG['n_imfs'])
        print(f"Loaded {len(imf_data['imfs'])} IMFs + residue")

    if args.decompose_only:
        print("\nDecomposition complete. Exiting.")
        return

    # STEP 2: Training
    if not args.evaluate_only:
        run_training(imf_data, device)

    if args.train_only:
        print("\nTraining complete. Exiting.")
        return

    # STEP 3: Evaluation
    results_df = run_full_evaluation(imf_data, target_data, device)

    # STEP 4: Generate series plots (same style as Baselines_model)
    print("\n" + "="*60)
    print("STEP 4: GENERATING PLOTS")
    print("="*60)
    plot_all('EC')

    # Print summary
    print_summary(results_df)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("  - metrics/full_evaluation_results.csv")
    print("  - metrics/summary_results.csv")
    print("  - series/*.csv (Actual vs Predicted)")
    print("  - plots/*.png (comparison plots)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CEEMDAN-based Water Quality Forecasting')
    parser.add_argument('--decompose-only', action='store_true',
                        help='Only run CEEMDAN decomposition')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train models (requires decomposed data)')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate (requires trained models)')
    parser.add_argument('--force-decompose', action='store_true',
                        help='Force re-run CEEMDAN even if cached')

    args = parser.parse_args()
    main(args)
