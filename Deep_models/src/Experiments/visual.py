"""
Visualization module for Deep Learning Models
Only uses PatchTST, LSTM, Transformer (no Linear models)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
from src.Utils.path import OUTPUT_DIR, DATA_DIR
from src.Model.patchtst import PatchTST
from src.Utils.parameter import CONFIG, device
from src.Utils.training import evaluate_model
from src.Data.data_loading import create_dataloaders_advanced

# Plot configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 150})
REPORT_DIR = OUTPUT_DIR / "report"
FIGURE_DIR = OUTPUT_DIR / "image"
os.makedirs(FIGURE_DIR, exist_ok=True)


def save_plot(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved plot: {path}")


def plot_model_comparison():
    csv_path = OUTPUT_DIR / "comparison_results.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping plot_model_comparison: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Filter out raw_data column if exists
    plot_cols = ['Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
    available_cols = [c for c in plot_cols if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAE
    if 'MAE' in df.columns:
        sns.barplot(data=df, x="Model", y="MAE", ax=axes[0, 0], palette="viridis")
        axes[0, 0].set_title("MAE Comparison", fontsize=12, fontweight='bold')
        for container in axes[0, 0].containers:
            axes[0, 0].bar_label(container, fmt='%.4f', padding=3)

    # RMSE
    if 'RMSE' in df.columns:
        sns.barplot(data=df, x="Model", y="RMSE", ax=axes[0, 1], palette="magma")
        axes[0, 1].set_title("RMSE Comparison", fontsize=12, fontweight='bold')
        for container in axes[0, 1].containers:
            axes[0, 1].bar_label(container, fmt='%.4f', padding=3)

    # MSE
    if 'MSE' in df.columns:
        sns.barplot(data=df, x="Model", y="MSE", ax=axes[1, 0], palette="rocket")
        axes[1, 0].set_title("MSE Comparison", fontsize=12, fontweight='bold')
        for container in axes[1, 0].containers:
            axes[1, 0].bar_label(container, fmt='%.4f', padding=3)

    # R2
    if 'R2' in df.columns:
        sns.barplot(data=df, x="Model", y="R2", ax=axes[1, 1], palette="crest")
        axes[1, 1].set_title("R² Comparison (Higher is Better)", fontsize=12, fontweight='bold')
        for container in axes[1, 1].containers:
            axes[1, 1].bar_label(container, fmt='%.4f', padding=3)

    save_plot("model_comparison.png")


def plot_runtime():
    csv_path = REPORT_DIR / "model_comparison_report.csv"
    if not os.path.exists(csv_path):
        # Try alternative path
        csv_path = OUTPUT_DIR / "comparison_results.csv"
        if not os.path.exists(csv_path):
            print("Skipping plot_runtime: No comparison file found")
            return

    df = pd.read_csv(csv_path)

    if 'Inference_ms' not in df.columns:
        print("Skipping plot_runtime: No Inference_ms column")
        return

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Model", y="Inference_ms", palette="magma")

    plt.title("Inference Speed Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Time per Sample (ms)")
    plt.xlabel("Model")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f ms', padding=3)

    save_plot("runtime_comparison.png")


def plot_horizon():
    csv_path = REPORT_DIR / "horizon_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping plot_horizon: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Check for possible column names
    horizon_col = None
    for col in ["Horizon", "horizon", "Pred_Len", "pred_len"]:
        if col in df.columns:
            horizon_col = col
            break

    if horizon_col is None:
        print(f"Skipping plot_horizon: No horizon column found. Columns: {df.columns.tolist()}")
        return

    plt.figure(figsize=(10, 6))

    if "MAE" in df.columns:
        sns.lineplot(data=df, x=horizon_col, y="MAE", marker="o", linewidth=2.5, label="MAE", color="blue")
    if "RMSE" in df.columns:
        sns.lineplot(data=df, x=horizon_col, y="RMSE", marker="s", linewidth=2.5, label="RMSE", color="red")
    if "MSE" in df.columns:
        sns.lineplot(data=df, x=horizon_col, y="MSE", marker="^", linewidth=2.5, label="MSE", color="green")

    plt.title("Forecasting Performance vs Prediction Horizon", fontsize=14, fontweight='bold')
    plt.xlabel("Prediction Horizon (Hours)")
    plt.ylabel("Error")
    plt.xticks(df[horizon_col].unique())
    plt.legend()
    plt.grid(True, linestyle='--')

    save_plot("horizon_analysis.png")


def plot_alpha():
    csv_path = REPORT_DIR / "alpha_sensitivity.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping plot_alpha: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    if "Alpha" not in df.columns or "MAE" not in df.columns:
        print("Skipping plot_alpha: Missing required columns")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Alpha", y="MAE", marker="o", markersize=10, color="green", linewidth=2.5)

    # Mark optimal point
    min_idx = df['MAE'].idxmin()
    best_alpha = df.loc[min_idx, "Alpha"]
    best_mae = df.loc[min_idx, "MAE"]
    plt.annotate(f'Optimal (α={best_alpha})', xy=(best_alpha, best_mae),
                 xytext=(best_alpha + 0.5, best_mae * 1.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10)

    plt.title("Impact of Event Weight (Alpha) on MAE", fontsize=14, fontweight='bold')
    plt.xlabel("Alpha (Event Weight)")
    plt.ylabel("MAE")
    plt.xticks(df["Alpha"].unique())
    plt.grid(True, linestyle='--')

    save_plot("alpha_sensitivity.png")


def plot_stability():
    csv_path = REPORT_DIR / "stability_report.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping plot_stability: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    if "MAE" not in df.columns:
        print("Skipping plot_stability: No MAE column")
        return

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df["MAE"], width=0.4, color="skyblue")
    sns.stripplot(y=df["MAE"], color="black", size=10, jitter=True)

    mean_val = df["MAE"].mean()
    std_val = df["MAE"].std()

    plt.title(f"Model Stability (5 Random Seeds)\nMean: {mean_val:.4f} ± {std_val:.4f}", fontsize=14)
    plt.ylabel("MAE Distribution")

    save_plot("stability_analysis.png")


def plot_patchtst_configs():
    csv_path = REPORT_DIR / "patchtst_configs.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping plot_patchtst_configs: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    if "Config" not in df.columns:
        print("Skipping plot_patchtst_configs: No Config column")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE by config
    if 'MAE' in df.columns:
        sns.barplot(data=df, x="Config", y="MAE", ax=axes[0], palette="viridis")
        axes[0].set_title("PatchTST: MAE by Configuration", fontsize=12, fontweight='bold')
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.4f', padding=3)

    # R2 by config
    if 'R2' in df.columns:
        sns.barplot(data=df, x="Config", y="R2", ax=axes[1], palette="crest")
        axes[1].set_title("PatchTST: R² by Configuration", fontsize=12, fontweight='bold')
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.4f', padding=3)

    save_plot("patchtst_configs.png")


def plot_all_targets_sample(y_true, y_pred, target_names, sample_idx=0):
    """Plot predictions vs actuals for all targets."""
    num_targets = len(target_names)

    # Adjust grid based on number of targets
    if num_targets <= 2:
        nrows, ncols = 1, 2
    elif num_targets <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 3, 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i in range(min(num_targets, len(axes))):
        true_series = y_true[sample_idx, :, i]
        pred_series = y_pred[sample_idx, :, i]

        ax = axes[i]
        ax.plot(true_series, label="Actual", color="#2ecc71", linewidth=2, marker='o', markersize=3)
        ax.plot(pred_series, label="Predicted", color="#e74c3c", linestyle="--", linewidth=2, marker='x', markersize=3)

        ax.set_title(f"Target: {target_names[i]}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Forecast Step (hours)")
        ax.set_ylabel("Value")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_targets, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    save_plot(f"all_targets_sample_{sample_idx}.png")


def plot_all():
    """Generate all available plots."""
    print("\nGENERATING PLOTS...")

    plot_model_comparison()
    plot_runtime()
    plot_horizon()
    plot_alpha()
    plot_stability()
    plot_patchtst_configs()

    # Try to generate sample prediction plots
    try:
        possible_paths = [
            OUTPUT_DIR / "Final_Processed_Data.csv",
            DATA_DIR / "USGs/water_data_2021_2025_clean.csv",
        ]

        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"Loaded data from: {path}")
                break

        if df is None:
            print("Skipping sample plots: No data file found")
            print(f"All plots saved to '{FIGURE_DIR}' folder.")
            return

        loaders, _, split_info = create_dataloaders_advanced(
            df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size']
        )
        train_loader, val_loader, test_loader = loaders

        # Try to load PatchTST model
        input_dim = split_info['n_features']
        output_dim = split_info['n_targets']

        model = PatchTST(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=CONFIG['seq_len'],
            pred_len=CONFIG['pred_len']
        )

        model_paths = [
            OUTPUT_DIR / "model/best_model_PatchTST_len24_alpha5.042.pth",
            OUTPUT_DIR / "model/best_model_PatchTST_len24_alpha1.042.pth",
        ]

        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model_loaded = True
                    print(f"Loaded model from: {model_path}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
                    continue

        if not model_loaded:
            print("Skipping sample plots: No compatible model found")
            print(f"All plots saved to '{FIGURE_DIR}' folder.")
            return

        model.to(device)
        model.eval()
        results = evaluate_model(model, test_loader, device, split_info)
        y_true = results['raw_data']['actuals']
        y_pred = results['raw_data']['preds']
        target_names = split_info.get('target_names', CONFIG['targets'])

        # Generate sample plots
        max_samples = len(y_true)
        sample_indices = [i for i in [0, 10, 50] if i < max_samples]
        for i in sample_indices:
            plot_all_targets_sample(y_true, y_pred, target_names, sample_idx=i)

    except Exception as e:
        print(f"Error generating sample plots: {e}")
        import traceback
        traceback.print_exc()

    print(f"All plots saved to '{FIGURE_DIR}' folder.")


if __name__ == "__main__":
    plot_all()
