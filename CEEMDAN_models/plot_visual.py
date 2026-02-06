"""
Visualization Module for CEEMDAN-based Forecasting
Generates series comparison plots (Actual vs Predicted)
Same style as Baselines_model
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import RESULTS_DIR


def plot_prediction(file_path: str, save_path: str):
    """
    Plot Actual vs Predicted comparison from CSV file.
    Same style as Baselines_model.

    Args:
        file_path: Path to CSV file with 'Actual' and 'Predicted' columns
        save_path: Path to save PNG image
    """
    df = pd.read_csv(file_path)

    plt.figure(figsize=(15, 5))

    # Plot Actual vs Predicted (same colors as Baselines_model)
    plt.plot(df['Actual'], label='Actual', color='blue', linewidth=1.5)
    plt.plot(df['Predicted'], label='Predicted', color='red', linestyle='--', linewidth=1.2)

    plt.title(f"Comparison: {Path(file_path).stem}", fontsize=14, fontweight='bold')
    plt.xlabel("Time steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all(target: str = 'EC'):
    """
    Generate plots for all series files containing the target variable.

    Args:
        target: Target variable name (e.g., 'EC')
    """
    series_dir = RESULTS_DIR / 'series'
    plots_dir = RESULTS_DIR / 'plots'

    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)

    if not series_dir.exists():
        print(f"Series directory not found: {series_dir}")
        return

    # Get all CSV files for this target
    filenames = [f for f in os.listdir(series_dir) if f.endswith('.csv') and target in f]

    if not filenames:
        print(f"No series files found for target '{target}' in {series_dir}")
        return

    print(f"\nGenerating {len(filenames)} plots for {target}...")

    for file in filenames:
        # Convert CSV filename to PNG
        png_name = file.replace('.csv', '.png')

        csv_path = series_dir / file
        png_path = plots_dir / png_name

        print(f"  {file} -> {png_name}")
        plot_prediction(str(csv_path), str(png_path))

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    # Generate plots for EC
    plot_all('EC')
