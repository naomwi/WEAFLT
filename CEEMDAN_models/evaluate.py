"""
Evaluation Module for CEEMDAN-based Forecasting
Evaluates trained models and reconstructs final predictions
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from config import (
    CEEMDAN_CONFIG, DATA_CONFIG, MODEL_DIR, RESULTS_DIR,
    HORIZONS, AVAILABLE_MODELS
)
from utils.data_loader import IMFDataManager, IMFDataset, load_raw_data, get_target_data, prepare_data_splits
from torch.utils.data import DataLoader
from utils.ceemdan_decomposition import load_imfs, reconstruct_from_predictions
from utils.metrics import calculate_all_metrics, print_metrics
from train import create_model, get_device


def load_trained_models(model_name: str, pred_len: int, device: torch.device, n_components: int = 13) -> Dict:
    """
    Load trained models for all components.

    Args:
        model_name: Name of model type
        pred_len: Prediction horizon
        device: Device to load models to
        n_components: Number of components (12 IMFs + 1 residue = 13)

    Returns:
        Dictionary of loaded models
    """
    models = {}
    seq_len = DATA_CONFIG['seq_len']

    # Load IMF models
    for i in range(n_components - 1):
        component_name = f'imf_{i+1}'
        model_path = MODEL_DIR / f"{model_name}_{component_name}_h{pred_len}.pth"

        if not model_path.exists():
            print(f"Warning: Model not found: {model_path}")
            continue

        model = create_model(model_name, seq_len, pred_len)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[component_name] = model

    # Load residue model
    component_name = 'residue'
    model_path = MODEL_DIR / f"{model_name}_{component_name}_h{pred_len}.pth"

    if model_path.exists():
        model = create_model(model_name, seq_len, pred_len)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[component_name] = model
    else:
        print(f"Warning: Residue model not found: {model_path}")

    return models


def evaluate_full_prediction(
    model_name: str,
    pred_len: int,
    imf_data: Dict,
    original_data: np.ndarray,
    device: torch.device
) -> Dict:
    """
    Evaluate full prediction by summing all component predictions.

    Args:
        model_name: Name of model type
        pred_len: Prediction horizon
        imf_data: Dictionary with 'imfs' and 'residue'
        original_data: Original (non-decomposed) target data
        device: Evaluation device

    Returns:
        Dictionary with evaluation results
    """
    n_imfs = CEEMDAN_CONFIG['n_imfs']
    seq_len = DATA_CONFIG['seq_len']
    batch_size = DATA_CONFIG['batch_size']

    # Load all trained models
    models = load_trained_models(model_name, pred_len, device, n_components=n_imfs + 1)

    if len(models) < n_imfs + 1:
        print(f"Warning: Only {len(models)} models loaded, expected {n_imfs + 1}")

    all_component_preds = []
    all_component_trues = []

    # Evaluate each IMF using IMFDataset (same as Baselines_model)
    for i in range(n_imfs):
        component_name = f'imf_{i+1}'

        if component_name not in models:
            print(f"Skipping {component_name}: model not found")
            continue

        model = models[component_name]
        component_data = imf_data['imfs'][i]

        # Create test dataset using IMFDataset (same split logic as Baselines_model)
        test_set = IMFDataset(component_data, seq_len, pred_len, flag='test')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # Get predictions
        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                pred = model(x)
                preds.append(pred.cpu().numpy())
                trues.append(y.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Lấy điểm cuối của mỗi prediction window (giống Baselines_model: [:, -1])
        preds_last = preds[:, -1, 0]  # Shape: (n_samples,)
        trues_last = trues[:, -1, 0]  # Shape: (n_samples,)

        # Inverse scale using test_set's scaler
        preds_flat = test_set.inverse(preds_last.reshape(-1, 1)).flatten()
        trues_flat = test_set.inverse(trues_last.reshape(-1, 1)).flatten()

        all_component_preds.append(preds_flat)
        all_component_trues.append(trues_flat)

    # Evaluate residue
    component_name = 'residue'
    if component_name in models:
        model = models[component_name]
        component_data = imf_data['residue']

        # Create test dataset using IMFDataset
        test_set = IMFDataset(component_data, seq_len, pred_len, flag='test')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                pred = model(x)
                preds.append(pred.cpu().numpy())
                trues.append(y.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Lấy điểm cuối của mỗi prediction window (giống Baselines_model)
        preds_last = preds[:, -1, 0]
        trues_last = trues[:, -1, 0]

        # Inverse scale using test_set's scaler
        preds_flat = test_set.inverse(preds_last.reshape(-1, 1)).flatten()
        trues_flat = test_set.inverse(trues_last.reshape(-1, 1)).flatten()

        residue_pred = preds_flat
        residue_true = trues_flat
    else:
        residue_pred = np.zeros_like(all_component_preds[0])
        residue_true = np.zeros_like(all_component_trues[0])

    # Reconstruct full prediction
    final_pred = reconstruct_from_predictions(all_component_preds, residue_pred)
    final_true = reconstruct_from_predictions(all_component_trues, residue_true)

    # Calculate metrics
    metrics = calculate_all_metrics(final_true, final_pred)

    return {
        'metrics': metrics,
        'predictions': final_pred,
        'actuals': final_true,
        'component_preds': all_component_preds,
        'residue_pred': residue_pred,
    }


def save_series_csv(
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    horizon: int,
    target: str = 'EC'
):
    """
    Save Actual vs Predicted series to CSV file.
    Same format as Baselines_model.

    Args:
        actuals: Actual values
        predictions: Predicted values
        model_name: Name of model
        horizon: Prediction horizon
        target: Target variable name
    """
    series_dir = RESULTS_DIR / 'series'
    os.makedirs(series_dir, exist_ok=True)

    # Same naming convention as Baselines_model
    filename = f"series_{model_name}_P{horizon}_{target}.csv"
    filepath = series_dir / filename

    df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions
    })
    df.to_csv(filepath, index=False)
    print(f"  Saved series: {filename}")


def run_evaluation(
    model_name: str,
    horizons: List[int] = None,
    imf_data: Dict = None,
    original_data: np.ndarray = None,
    device: torch.device = None,
    save_series: bool = True
) -> pd.DataFrame:
    """
    Run evaluation across all horizons.

    Args:
        model_name: Name of model type
        horizons: List of prediction horizons
        imf_data: IMF decomposition data
        original_data: Original target data
        device: Evaluation device
        save_series: Whether to save series CSV files

    Returns:
        DataFrame with evaluation results
    """
    if horizons is None:
        horizons = HORIZONS

    if device is None:
        device = get_device()

    results = []

    for h in horizons:
        print(f"\nEvaluating {model_name} at horizon {h}...")

        try:
            eval_result = evaluate_full_prediction(
                model_name, h, imf_data, original_data, device
            )

            metrics = eval_result['metrics']
            metrics['Model'] = model_name
            metrics['Horizon'] = h
            results.append(metrics)

            print_metrics(metrics, f"{model_name} (h={h})")

            # Save series CSV (same as Baselines_model)
            if save_series:
                save_series_csv(
                    eval_result['actuals'],
                    eval_result['predictions'],
                    model_name,
                    h,
                    target='EC'
                )

        except Exception as e:
            print(f"Error evaluating {model_name} at horizon {h}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)


def plot_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    model_name: str,
    horizon: int,
    save_path: Path = None,
    n_points: int = 500
):
    """
    Plot predictions vs actuals.

    Args:
        predictions: Predicted values
        actuals: Actual values
        model_name: Name of model
        horizon: Prediction horizon
        save_path: Path to save plot
        n_points: Number of points to plot
    """
    plt.figure(figsize=(12, 5))

    # Limit points for visibility
    n = min(n_points, len(predictions))
    idx = np.arange(n)

    plt.plot(idx, actuals[:n], label='Actual', color='blue', linewidth=1.5)
    plt.plot(idx, predictions[:n], label='Predicted', color='red', linewidth=1.5, alpha=0.8)

    plt.title(f'{model_name} Predictions (Horizon={horizon})', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('EC Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    plt.close()


def generate_comparison_report(results_df: pd.DataFrame, save_dir: Path = None):
    """
    Generate comparison plots and report.

    Args:
        results_df: DataFrame with all evaluation results
        save_dir: Directory to save reports
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    os.makedirs(save_dir / 'plots', exist_ok=True)
    os.makedirs(save_dir / 'metrics', exist_ok=True)

    # Save full results
    results_df.to_csv(save_dir / 'metrics' / 'full_evaluation_results.csv', index=False)

    # Plot metrics by horizon for each model
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'R2']

    for metric in metrics_to_plot:
        if metric not in results_df.columns:
            continue

        plt.figure(figsize=(10, 6))

        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            plt.plot(model_data['Horizon'], model_data[metric], marker='o', label=model, linewidth=2)

        plt.title(f'{metric} vs Prediction Horizon', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Horizon (hours)')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_dir / 'plots' / f'{metric.lower()}_by_horizon.png', dpi=150)
        plt.close()

    # Bar plot comparison at each horizon
    for horizon in results_df['Horizon'].unique():
        horizon_data = results_df[results_df['Horizon'] == horizon]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for ax, metric in zip(axes.flatten(), ['MAE', 'RMSE', 'MAPE', 'R2']):
            if metric in horizon_data.columns:
                ax.bar(horizon_data['Model'], horizon_data[metric], color='steelblue')
                ax.set_title(f'{metric} (Horizon={horizon})', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric)

                # Add value labels
                for i, v in enumerate(horizon_data[metric]):
                    ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_dir / 'plots' / f'comparison_h{horizon}.png', dpi=150)
        plt.close()

    print(f"\nReports saved to: {save_dir}")


if __name__ == "__main__":
    print("Testing evaluation module...")

    device = get_device()

    # This would require trained models - just testing imports
    print("Evaluation module loaded successfully!")
