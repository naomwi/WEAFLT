"""
Cross-Model Comparison Script
Compare metrics across all models: Proposed, CEEMD_Baselines, Deep_Baselines
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Project root
PROJECT_DIR = Path(__file__).parent

# Model folders
MODEL_FOLDERS = {
    'CA-CEEMDAN-DLinear': ('Proposed_Model', 'dlinear'),
    'CA-CEEMDAN-NLinear': ('Proposed_Model', 'nlinear'),
    'CEEMD-DLinear': ('CEEMD_Baselines', 'dlinear'),
    'CEEMD-NLinear': ('CEEMD_Baselines', 'nlinear'),
    'LSTM': ('Deep_Baselines', 'lstm'),
    'PatchTST': ('Deep_Baselines', 'patchtst'),
    'Transformer': ('Deep_Baselines', 'transformer'),
}

HORIZONS = [6, 12, 24, 48, 96, 168]
METRICS = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'MAE_Sudden']

# Color palette for models
MODEL_COLORS = {
    'CA-CEEMDAN-DLinear': '#1f77b4',  # Blue (Proposed)
    'CA-CEEMDAN-NLinear': '#17becf',  # Cyan (Proposed)
    'CEEMD-DLinear': '#2ca02c',       # Green
    'CEEMD-NLinear': '#98df8a',       # Light green
    'LSTM': '#ff7f0e',                # Orange
    'PatchTST': '#d62728',            # Red
    'Transformer': '#9467bd',         # Purple
}


def load_all_results(target: str = 'EC') -> pd.DataFrame:
    """
    Load metrics from all model folders.

    Args:
        target: 'EC' or 'pH'

    Returns:
        DataFrame with columns: Model, Horizon, MAE, RMSE, MAPE, R2, MAE_Sudden
    """
    all_results = []

    for model_name, (folder, model_type) in MODEL_FOLDERS.items():
        metrics_dir = PROJECT_DIR / folder / target / "results" / "metrics"

        if not metrics_dir.exists():
            print(f"  Warning: {metrics_dir} not found")
            continue

        for horizon in HORIZONS:
            csv_path = metrics_dir / f"{model_type}_h{horizon}.csv"

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['Model_Name'] = model_name
                df['Horizon'] = horizon
                all_results.append(df)
            else:
                print(f"  Warning: {csv_path} not found")

    if not all_results:
        print("No results found!")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    target: str,
    output_dir: Path,
    figsize: tuple = (14, 6)
) -> None:
    """
    Plot bar chart comparing a metric across all models and horizons.
    """
    if df.empty or metric not in df.columns:
        print(f"  Skipping {metric}: no data")
        return

    fig, ax = plt.subplots(figsize=figsize)

    models = list(MODEL_FOLDERS.keys())
    n_models = len(models)
    n_horizons = len(HORIZONS)

    bar_width = 0.8 / n_models
    x = np.arange(n_horizons)

    for i, model in enumerate(models):
        model_data = df[df['Model_Name'] == model]
        values = []

        for h in HORIZONS:
            h_data = model_data[model_data['Horizon'] == h]
            if len(h_data) > 0:
                values.append(h_data[metric].values[0])
            else:
                values.append(0)

        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=model, color=MODEL_COLORS.get(model, 'gray'),
                      alpha=0.8, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric} across Horizons ({target})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = output_dir / f"model_comparison_{metric}_{target}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_heatmap(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    metric: str = 'MAE'
) -> None:
    """
    Create heatmap of metric values (models x horizons).
    """
    if df.empty:
        return

    models = list(MODEL_FOLDERS.keys())
    pivot_data = []

    for model in models:
        model_data = df[df['Model_Name'] == model]
        row = []
        for h in HORIZONS:
            h_data = model_data[model_data['Horizon'] == h]
            if len(h_data) > 0 and metric in h_data.columns:
                row.append(h_data[metric].values[0])
            else:
                row.append(np.nan)
        pivot_data.append(row)

    pivot_df = pd.DataFrame(pivot_data, index=models, columns=HORIZONS)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(pivot_df.values, cmap='RdYlGn_r', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric, rotation=-90, va="bottom", fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(HORIZONS)
    ax.set_yticklabels(models)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add value annotations
    for i in range(len(models)):
        for j in range(len(HORIZONS)):
            val = pivot_df.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}',
                              ha="center", va="center", color="black", fontsize=8)

    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(f'{metric} Heatmap - All Models ({target})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = output_dir / f"model_comparison_heatmap_{metric}_{target}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_summary_table(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    """
    Generate summary CSV table.
    """
    if df.empty:
        return

    # Pivot table: rows=models, columns=horizons, values=MAE
    summary_rows = []

    for model in MODEL_FOLDERS.keys():
        model_data = df[df['Model_Name'] == model]
        row = {'Model': model}

        for h in HORIZONS:
            h_data = model_data[model_data['Horizon'] == h]
            if len(h_data) > 0:
                row[f'MAE_H{h}'] = h_data['MAE'].values[0]
                row[f'RMSE_H{h}'] = h_data['RMSE'].values[0]
                row[f'R2_H{h}'] = h_data['R2'].values[0]
            else:
                row[f'MAE_H{h}'] = np.nan
                row[f'RMSE_H{h}'] = np.nan
                row[f'R2_H{h}'] = np.nan

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    save_path = output_dir / f"model_comparison_summary_{target}.csv"
    summary_df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


def plot_best_model_ranking(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    """
    Plot ranking of models (lower MAE = better rank).
    """
    if df.empty:
        return

    # Calculate average MAE across all horizons
    avg_metrics = df.groupby('Model_Name').agg({
        'MAE': 'mean',
        'RMSE': 'mean',
        'R2': 'mean'
    }).reset_index()

    avg_metrics = avg_metrics.sort_values('MAE')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE ranking
    colors = [MODEL_COLORS.get(m, 'gray') for m in avg_metrics['Model_Name']]
    axes[0].barh(avg_metrics['Model_Name'], avg_metrics['MAE'], color=colors, alpha=0.8)
    axes[0].set_xlabel('Average MAE')
    axes[0].set_title('MAE Ranking (lower is better)')
    axes[0].invert_yaxis()

    # RMSE ranking
    avg_metrics_rmse = avg_metrics.sort_values('RMSE')
    colors_rmse = [MODEL_COLORS.get(m, 'gray') for m in avg_metrics_rmse['Model_Name']]
    axes[1].barh(avg_metrics_rmse['Model_Name'], avg_metrics_rmse['RMSE'], color=colors_rmse, alpha=0.8)
    axes[1].set_xlabel('Average RMSE')
    axes[1].set_title('RMSE Ranking (lower is better)')
    axes[1].invert_yaxis()

    # R2 ranking
    avg_metrics_r2 = avg_metrics.sort_values('R2', ascending=False)
    colors_r2 = [MODEL_COLORS.get(m, 'gray') for m in avg_metrics_r2['Model_Name']]
    axes[2].barh(avg_metrics_r2['Model_Name'], avg_metrics_r2['R2'], color=colors_r2, alpha=0.8)
    axes[2].set_xlabel('Average R²')
    axes[2].set_title('R² Ranking (higher is better)')
    axes[2].invert_yaxis()

    plt.suptitle(f'Model Rankings - {target}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f"model_ranking_{target}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare models across all experiments')
    parser.add_argument('--target', '-t', default='EC', choices=['EC', 'pH', 'both'],
                        help='Target variable')
    args = parser.parse_args()

    # Output directory
    output_dir = PROJECT_DIR / "results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = ['EC', 'pH'] if args.target == 'both' else [args.target]

    for target in targets:
        print(f"\n{'='*60}")
        print(f"Generating comparison plots for {target}")
        print(f"{'='*60}")

        # Load results
        print("\nLoading results...")
        df = load_all_results(target)

        if df.empty:
            print(f"No results found for {target}")
            continue

        print(f"  Loaded {len(df)} result rows")

        # Generate plots
        print("\nGenerating comparison plots...")
        for metric in ['MAE', 'RMSE', 'R2', 'MAE_Sudden']:
            plot_metric_comparison(df, metric, target, output_dir)

        print("\nGenerating heatmaps...")
        for metric in ['MAE', 'RMSE']:
            plot_heatmap(df, target, output_dir, metric)

        print("\nGenerating summary table...")
        generate_summary_table(df, target, output_dir)

        print("\nGenerating ranking plots...")
        plot_best_model_ranking(df, target, output_dir)

    print(f"\n{'='*60}")
    print(f"All comparison plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
