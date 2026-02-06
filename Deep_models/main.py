"""
Deep Learning Models for Water Quality Forecasting
Includes: PatchTST, LSTM, Transformer

This script runs deep learning model comparison experiments.
"""

import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from src.Utils.parameter import CONFIG, RUNTIME_LOG, device
from src.seed import seed_everything
from src.Data.Data_processor import DataProcessor
from src.Utils.path import OUTPUT_DIR, DATA_DIR
from src.Experiments.deploy_experiments import (
    exp_model_comparison,
    exp_alpha_sensitivity,
    exp_horizon_comparison,
    exp_stability,
    exp_patchtst_configs
)
from src.Experiments.visual import plot_all

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


def data_processing():
    """Load and preprocess data with CEEMDAN decomposition."""
    possible_paths = [
        DATA_DIR / 'USGs/water_data_2021_2025_clean.csv',
        DATA_DIR / 'New_data/USGS/water_data_2021_2025_clean.csv',
    ]

    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        raise FileNotFoundError(f"USGS data not found. Tried: {possible_paths}")

    print(f">>> Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    processor = DataProcessor(CONFIG)
    df_final = processor.run_pipeline(df)

    cols = df_final.columns
    print(f"Total columns: {len(cols)}")
    print("Sample Columns:", cols[:5].tolist())

    # Save processed data
    output_path = OUTPUT_DIR / "Final_Processed_Data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f">>> Saved processed data to: {output_path}")

    return df_final


def main():
    print("=" * 80)
    print("DEEP LEARNING MODELS - WATER QUALITY FORECASTING")
    print("Models: PatchTST, LSTM, Transformer")
    print("=" * 80)

    seed_everything(42)

    df_main = data_processing()

    print("\nRUNNING EXPERIMENTS")
    print("=" * 80)

    # 1. Model Comparison (PatchTST vs LSTM vs Transformer)
    print("--> MODEL COMPARISON <--")
    exp_model_comparison(df_main)
    print("=" * 80)

    # 2. Alpha Sensitivity (using PatchTST)
    print("--> ALPHA SENSITIVITY <--")
    exp_alpha_sensitivity(df_main)
    print("=" * 80)

    # 3. Stability Test
    print("--> STABILITY TEST <--")
    exp_stability(df_main)
    print("=" * 80)

    # 4. Horizon Comparison
    print("--> HORIZON COMPARISON <--")
    exp_horizon_comparison(df_main)
    print("=" * 80)

    # 5. PatchTST Configuration Analysis
    print("--> PATCHTST CONFIG ANALYSIS <--")
    exp_patchtst_configs(df_main)
    print("=" * 80)

    # Save runtime log
    print("Saving runtime log...")
    df_runtime = pd.DataFrame(RUNTIME_LOG)
    df_runtime.to_csv(OUTPUT_DIR / "report/runtime_report.csv", index=False)

    # Generate plots
    print("Generating plots...")
    try:
        plot_all()
    except Exception as e:
        print(f"Warning: Plot generation failed: {e}")

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("  - comparison_results.csv")
    print("  - report/model_comparison_report.csv")
    print("  - report/alpha_sensitivity.csv")
    print("  - report/stability_report.csv")
    print("  - report/horizon_comparison.csv")
    print("  - report/patchtst_configs.csv")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'report', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'image', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'model', exist_ok=True)
    main()
