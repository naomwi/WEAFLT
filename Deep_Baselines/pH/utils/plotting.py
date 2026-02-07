"""
Visualization Module for CA-CEEMDAN-LTSF
Generates series comparison plots (Actual vs Predicted)
Target: EC
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional


def plot_prediction(
    actual: Union[np.ndarray, list],
    predicted: Union[np.ndarray, list],
    title: str,
    save_path: Union[str, Path],
    figsize: tuple = (20, 6),
    dpi: int = 150
) -> None:
    """
    Plot Actual vs Predicted comparison.

    Args:
        actual: Ground truth values
        predicted: Model predictions
        title: Plot title
        save_path: Path to save PNG image
        figsize: Figure size (width, height)
        dpi: Resolution for saved image
    """
    plt.figure(figsize=figsize)

    # Plot Actual vs Predicted
    plt.plot(actual, label='Actual', color='blue', linewidth=1.5)
    plt.plot(predicted, label='Predicted', color='red', linestyle='--', linewidth=1.2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time steps")
    plt.ylabel("Value")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Ensure directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_from_csv(
    file_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (20, 6),
    dpi: int = 150
) -> None:
    """
    Plot Actual vs Predicted from CSV file.

    Args:
        file_path: Path to CSV file with 'Actual' and 'Predicted' columns
        save_path: Path to save PNG image (default: same as CSV with .png extension)
        figsize: Figure size
        dpi: Resolution
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    if save_path is None:
        save_path = file_path.with_suffix('.png')

    title = f"Comparison: {file_path.stem}"
    plot_prediction(df['Actual'].values, df['Predicted'].values, title, save_path, figsize, dpi)


def plot_all_series(
    results_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "series_*.csv"
) -> None:
    """
    Batch process all series CSV files in directory.

    Args:
        results_dir: Directory containing series CSV files
        output_dir: Directory to save plots (default: results_dir/plots)
        pattern: Glob pattern for CSV files
    """
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = results_dir / 'plots'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(results_dir.glob(pattern))

    if not csv_files:
        print(f"No CSV files matching '{pattern}' found in {results_dir}")
        return

    print(f"\nGenerating {len(csv_files)} plots...")

    for csv_path in csv_files:
        png_path = output_dir / csv_path.with_suffix('.png').name
        plot_from_csv(csv_path, png_path)

    print(f"All plots saved to: {output_dir}")


def plot_metrics_by_horizon(
    metrics_df: pd.DataFrame,
    metric_name: str,
    title: str,
    save_path: Union[str, Path],
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot a specific metric across different horizons.

    Args:
        metrics_df: DataFrame with 'horizon' column and metric columns
        metric_name: Name of metric to plot
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    plt.bar(metrics_df['horizon'].astype(str), metrics_df[metric_name],
            color='steelblue', alpha=0.8, edgecolor='navy')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Forecast Horizon (hours)")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test with dummy data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import RESULTS_DIR

    print("Plotting module loaded successfully")
    print(f"Results directory: {RESULTS_DIR}")
