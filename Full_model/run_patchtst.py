"""
Standalone PatchTST training and evaluation script.

This script runs only the PatchTST model for water quality forecasting.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.Utils.parameter import CONFIG, device
from src.Utils.path import OUTPUT_DIR, DATA_DIR
from src.Utils.support_class import EventWeightedMSE
from src.Utils.training import train_model, evaluate_model
from src.Data.data_loading import create_dataloaders_advanced
from src.Data.Data_processor import DataProcessor
from src.Model.patchtst import PatchTST
from src.seed import seed_everything

warnings.filterwarnings("ignore")


def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")
    print("=" * 80)


def load_or_process_data():
    """Load processed data or run preprocessing pipeline."""
    processed_path = OUTPUT_DIR / "Final_Processed_Data.csv"

    if processed_path.exists():
        print(f">>> Loading existing processed data from: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    # Process data if not exists
    print(">>> Processing data from scratch...")
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

    print(f">>> Loading raw data from: {csv_path}")
    df = pd.read_csv(csv_path)

    processor = DataProcessor(CONFIG)
    df_final = processor.run_pipeline(df)

    df_final.to_csv(processed_path, index=False)
    print(f">>> Saved processed data to: {processed_path}")

    return df_final


def run_patchtst_experiment(df, patch_configs=None, force_train=False):
    """
    Run PatchTST experiments with different configurations.

    Args:
        df: Processed dataframe
        patch_configs: List of patch configurations to test
        force_train: If True, retrain even if checkpoint exists

    Returns:
        DataFrame with results
    """
    if patch_configs is None:
        patch_configs = [
            {'patch_len': 16, 'stride': 8, 'label': 'P16_S8_default'},
        ]

    print("\n" + "=" * 80)
    print("PATCHTST EXPERIMENTS")
    print("=" * 80)

    # Create dataloaders
    loaders, _, info = create_dataloaders_advanced(
        df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size']
    )
    train_loader, val_loader, test_loader = loaders

    in_dim = info['n_features']
    out_dim = info['n_targets']

    print(f"\nDataset Info:")
    print(f"  Input features: {in_dim}")
    print(f"  Output targets: {out_dim}")
    print(f"  Sequence length: {CONFIG['seq_len']}")
    print(f"  Prediction horizon: {CONFIG['pred_len']}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    results = []

    for config in patch_configs:
        patch_len = config['patch_len']
        stride = config['stride']
        label = config['label']
        d_model = config.get('d_model', 128)
        nhead = config.get('nhead', 8)
        num_layers = config.get('num_layers', 3)

        print(f"\n{'=' * 60}")
        print(f"Testing PatchTST Configuration: {label}")
        print(f"  patch_len={patch_len}, stride={stride}")
        print(f"  d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        print(f"{'=' * 60}")

        try:
            # Create model
            model = PatchTST(
                input_dim=in_dim,
                output_dim=out_dim,
                seq_len=CONFIG['seq_len'],
                pred_len=CONFIG['pred_len'],
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=0.1,
                use_revin=True
            ).to(device)

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Model parameters: {n_params:,}")

            # Setup training
            criterion = EventWeightedMSE(alpha=CONFIG['event_weight']).to(device)
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

            model_name = f"PatchTST_{label}_len{CONFIG['pred_len']}_alpha{CONFIG['event_weight']}"

            # Train
            start_time = time.time()
            model = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                CONFIG['epochs'], device, model_name=model_name,
                force_train=force_train
            )
            train_time = time.time() - start_time

            # Evaluate
            print("\nEvaluating on test set...")
            metrics = evaluate_model(model, test_loader, device, info)

            # Measure inference time
            model.eval()
            inference_times = []
            with torch.no_grad():
                for x, _, _, _ in test_loader:
                    x = x.to(device)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t_start = time.time()
                    _ = model(x)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    inference_times.append((time.time() - t_start) / x.size(0))

            metrics['Config'] = label
            metrics['Patch_Len'] = patch_len
            metrics['Stride'] = stride
            metrics['D_Model'] = d_model
            metrics['N_Heads'] = nhead
            metrics['N_Layers'] = num_layers
            metrics['Params'] = n_params
            metrics['Train_Time_s'] = train_time
            metrics['Inference_ms'] = np.mean(inference_times) * 1000

            results.append(metrics)

            print(f"\nResults for {label}:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  R2: {metrics['R2']:.4f}")
            print(f"  SF_MAE: {metrics['SF_MAE']:.4f}")
            print(f"  Inference: {metrics['Inference_ms']:.2f} ms/sample")

        except Exception as e:
            print(f"ERROR running {label}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if results:
        df_results = pd.DataFrame(results)

        # Reorder columns
        col_order = ['Config', 'Patch_Len', 'Stride', 'D_Model', 'N_Heads', 'N_Layers',
                     'RMSE', 'MAE', 'R2', 'MAPE', 'SF_MAE', 'N_Events',
                     'Precision', 'Recall', 'F1_Score', 'AUROC',
                     'Params', 'Train_Time_s', 'Inference_ms']
        available_cols = [c for c in col_order if c in df_results.columns]
        other_cols = [c for c in df_results.columns if c not in col_order and c != 'raw_data']
        df_results = df_results[available_cols + other_cols]

        output_path = OUTPUT_DIR / "report/patchtst_standalone_results.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\n>>> Results saved to: {output_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("PATCHTST EXPERIMENT SUMMARY")
        print("=" * 80)
        summary_cols = ['Config', 'RMSE', 'MAE', 'SF_MAE', 'Inference_ms', 'Params']
        print(df_results[[c for c in summary_cols if c in df_results.columns]].to_string(index=False))
        print("=" * 80)

        return df_results

    return None


def main():
    """Main entry point."""
    print("=" * 80)
    print("PATCHTST WATER QUALITY FORECASTING")
    print("=" * 80)

    print_gpu_info()

    # Set seed for reproducibility
    seed_everything(42)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'report', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'model', exist_ok=True)
    os.makedirs(OUTPUT_DIR / 'image', exist_ok=True)

    # Load or process data
    df = load_or_process_data()

    # Define patch configurations to test
    patch_configs = [
        # Default configuration
        {'patch_len': 16, 'stride': 8, 'label': 'P16_S8_default', 'd_model': 128, 'nhead': 8, 'num_layers': 3},

        # Different patch sizes
        {'patch_len': 8, 'stride': 4, 'label': 'P8_S4', 'd_model': 128, 'nhead': 8, 'num_layers': 3},
        {'patch_len': 24, 'stride': 12, 'label': 'P24_S12', 'd_model': 128, 'nhead': 8, 'num_layers': 3},
        {'patch_len': 32, 'stride': 16, 'label': 'P32_S16', 'd_model': 128, 'nhead': 8, 'num_layers': 3},
    ]

    # Run experiments (set force_train=True to retrain from scratch)
    results = run_patchtst_experiment(df, patch_configs, force_train=False)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
