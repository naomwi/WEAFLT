import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.Utils.parameter import CONFIG,RUNTIME_LOG,device
from src.seed import seed_everything
from src.Utils.support_class import EarlyStopping
from src.Data.Data_processor import DataProcessor 
from src.Utils.path import OUTPUT_DIR,DATA_DIR
from src.Experiments.deploy_experiments import (
    exp_ablation_study,
    exp_alpha_sensitivity,
    exp_horizon_comparison,
    exp_model_comparison,
    exp_stability,
    exp_xai_analysis,
    exp_full_baseline_comparison,
    exp_patchtst_analysis,
    exp_ceemdan_variants,
    exp_imf_pruning,
    run_all_experiments
)
from src.Experiments.visual import plot_all
# --- config ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


# --- preprocess data ---
def data_processing():
    # Try multiple paths for USGS data
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
    print("Sample Residue:", [c for c in cols if 'residue' in c][:2])
    print("Sample Event:", [c for c in cols if 'event_flag' in c][:2])

    # Save processed data
    output_path = OUTPUT_DIR / "Final_Processed_Data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f">>> Saved processed data to: {output_path}")

    return df_final

def main(run_all=True, quick_mode=False):
    """
    Main entry point for all experiments.

    Args:
        run_all: If True, run all experiments including PatchTST, LSTM, Transformer
        quick_mode: If True, only run basic model comparison (faster)
    """
    print("="*80)
    print("WATER QUALITY FORECASTING - CEEMDAN-Enhanced LTSF-Linear")
    print("Models: DLinear, NLinear, RLinear, PatchTST, LSTM, Transformer")
    print("="*80)

    seed_everything(42)

    df_main = data_processing()

    print("\nRUNNING EXPERIMENTS")
    print("="*80)

    if quick_mode:
        # Quick mode: only basic comparison
        print("--> QUICK MODE: Basic Model Comparison <--")
        exp_model_comparison(df_main, include_all_baselines=False)
    elif run_all:
        # Run ALL experiments using the convenience function
        run_all_experiments(df_main)
    else:
        # Standard experiments
        print("--> [1/10] MODEL COMPARISON (All Baselines) <--")
        exp_model_comparison(df_main, include_all_baselines=True)
        print("="*80)

        print("--> [2/10] ALPHA SENSITIVITY <--")
        exp_alpha_sensitivity(df_main)
        print("="*80)

        print("--> [3/10] STABILITY (5 Seeds) <--")
        exp_stability(df_main)
        print("="*80)

        print("--> [4/10] ABLATION STUDY <--")
        exp_ablation_study(df_main)
        print("="*80)

        print("--> [5/10] HORIZON COMPARISON <--")
        exp_horizon_comparison(df_main)
        print("="*80)

        print("--> [6/10] XAI ANALYSIS <--")
        exp_xai_analysis(df_main)
        print("="*80)

        print("--> [7/10] FULL BASELINE COMPARISON (incl. PatchTST) <--")
        exp_full_baseline_comparison(df_main)
        print("="*80)

        print("--> [8/10] PATCHTST CONFIGURATION ANALYSIS <--")
        exp_patchtst_analysis(df_main)
        print("="*80)

        print("--> [9/10] CEEMDAN VARIANTS COMPARISON <--")
        exp_ceemdan_variants(df_main)
        print("="*80)

        print("--> [10/10] IMF PRUNING ANALYSIS <--")
        exp_imf_pruning(df_main)
        print("="*80)

    # Save runtime log
    print("\nSaving Runtime Log...")
    if RUNTIME_LOG:
        df_runtime = pd.DataFrame(RUNTIME_LOG)
        df_runtime.to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)

    # Generate plots
    print("\nGenerating Plots...")
    print("="*80)
    try:
        plot_all()
    except Exception as e:
        print(f"Plot generation error: {e}")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS FINISHED!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Water Quality Forecasting Experiments')
    parser.add_argument('--quick', action='store_true', help='Quick mode: only basic Linear models')
    parser.add_argument('--all', action='store_true', default=True, help='Run all experiments (default)')
    parser.add_argument('--standard', action='store_true', help='Run standard experiments without extended baselines')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR/'report', exist_ok=True)
    os.makedirs(OUTPUT_DIR/'image', exist_ok=True)
    os.makedirs(OUTPUT_DIR/'model', exist_ok=True)

    # Determine run mode
    if args.quick:
        main(run_all=False, quick_mode=True)
    elif args.standard:
        main(run_all=False, quick_mode=False)
    else:
        main(run_all=True, quick_mode=False)
