import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.Data.data_loading import create_dataloaders_advanced
from src.Model.patchtst import PatchTST
from src.Model.baselines.lstm import LSTMModel
from src.Model.baselines.transformer import TransformerModel
from src.Utils.parameter import CONFIG, device
from src.Utils.path import OUTPUT_DIR
from src.Utils.support_class import EventWeightedMSE
from src.Utils.training import evaluate_model, train_model
from src.seed import seed_everything

# Model registry - Only deep learning models
MODEL_REGISTRY = {
    "PatchTST": PatchTST,
    "LSTM": LSTMModel,
    "Transformer": TransformerModel,
}


def get_model(model_name, input_dim, output_dim, seq_len, pred_len, device, **kwargs):
    """Factory function to create models by name."""
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_class = MODEL_REGISTRY[model_name]

    if model_name == "PatchTST":
        patch_len = kwargs.get('patch_len', 16)
        stride = kwargs.get('stride', 8)
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 3)
        model = model_class(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_name == "LSTM":
        model = model_class(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=kwargs.get('hidden_size', 128),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_name == "Transformer":
        model = model_class(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=kwargs.get('d_model', 128),
            nhead=kwargs.get('nhead', 8),
            num_encoder_layers=kwargs.get('num_encoder_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        model = model_class(input_dim, output_dim, seq_len, pred_len)

    return model.to(device)


def run_single_trial(loaders, split_info, model_name, override_config=None, name_suffix=None):
    """Run a single training and evaluation trial."""
    cfg = CONFIG.copy()
    if override_config:
        cfg.update(override_config)

    train_loader, val_loader, test_loader = loaders
    in_dim, out_dim = split_info['n_features'], split_info['n_targets']

    model = get_model(model_name, in_dim, out_dim, cfg['seq_len'], cfg['pred_len'], device)

    criterion = EventWeightedMSE(alpha=cfg['event_weight']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    if not name_suffix:
        name_suffix = cfg.get('seed', 42)
    unique_name = f"{model_name}_len{cfg['pred_len']}_alpha{cfg['event_weight']}{name_suffix}"

    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                        cfg['epochs'], device, model_name=unique_name)

    # Measure Inference Time
    model.eval()
    times = []
    metrics = {}

    if test_loader:
        with torch.no_grad():
            for x, _, _, _ in test_loader:
                x = x.to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                st = time.time()
                _ = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.time() - st) / x.size(0))

        metrics = evaluate_model(model, test_loader, device, split_info)
        metrics['Inference_ms'] = np.mean(times) * 1000

    return metrics


def exp_model_comparison(df):
    """Compare PatchTST, LSTM, and Transformer."""
    print("\n[EXP 1] Deep Learning Model Comparison...")

    loaders, _, info = create_dataloaders_advanced(
        df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size']
    )

    models_to_compare = ["PatchTST", "LSTM", "Transformer"]

    results = []
    for m in models_to_compare:
        print(f"   Running {m}...")
        try:
            start_time = time.time()
            metrics = run_single_trial(loaders, info, m)
            metrics['Model'] = m
            metrics['Train_Time_s'] = time.time() - start_time
            results.append(metrics)
            print(f"      {m}: MSE={metrics.get('MSE', 'N/A'):.4f}, RMSE={metrics.get('RMSE', 'N/A'):.4f}, R2={metrics.get('R2', 'N/A'):.4f}")
        except Exception as e:
            print(f"   Error running {m}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        df_res = pd.DataFrame(results)

        # Reorder columns
        col_order = ['Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'SF_MAE', 'Inference_ms', 'Train_Time_s']
        available_cols = [c for c in col_order if c in df_res.columns]
        other_cols = [c for c in df_res.columns if c not in col_order and c != 'raw_data']
        df_res = df_res[available_cols + other_cols]

        df_res.to_csv(f"{OUTPUT_DIR}/comparison_results.csv", index=False)
        df_res[['Model', 'Inference_ms']].to_csv(OUTPUT_DIR / "report/model_comparison_report.csv", index=False)
        print("Model Comparison report SAVED!")
    else:
        print("Warning: No models completed successfully.")

    return results


def exp_alpha_sensitivity(df):
    """Test different alpha (event weight) values."""
    print("\n[EXP 2] Alpha Sensitivity...")

    loaders, _, info = create_dataloaders_advanced(
        df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size']
    )

    results = []
    alphas = [1.0, 3.0, 5.0, 10.0]

    for alpha in alphas:
        print(f"   Alpha = {alpha}...")
        metrics = run_single_trial(loaders, info, "PatchTST", override_config={'event_weight': alpha})
        metrics['Alpha'] = alpha
        results.append(metrics)

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "report/alpha_sensitivity.csv", index=False)
    print("Alpha sensitivity SAVED!")


def exp_horizon_comparison(df):
    """Test different prediction horizons."""
    print("\n[EXP 3] Forecasting Horizon...")

    results = []
    horizons = [24, 48, 96, 192]

    for h in horizons:
        print(f"   Horizon = {h}...")
        loaders, _, info = create_dataloaders_advanced(
            df, CONFIG['targets'], CONFIG['seq_len'],
            pred_len=h,
            batch_size=CONFIG['batch_size']
        )

        metrics = run_single_trial(loaders, info, "PatchTST", override_config={'pred_len': h})
        metrics['Horizon'] = h
        results.append(metrics)

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "report/horizon_comparison.csv", index=False)
    print("Horizon comparison SAVED!")


def exp_stability(df):
    """Test model stability across different seeds."""
    print("\n[EXP 4] Stability Test...")

    loaders, _, info = create_dataloaders_advanced(
        df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size']
    )

    results = []
    seeds = [42, 100, 2024, 777, 99]

    for s in seeds:
        print(f"   Seed = {s}...")
        seed_everything(s)
        metrics = run_single_trial(loaders, info, "PatchTST", name_suffix=f"_seed{s}")
        metrics['Seed'] = s
        results.append(metrics)

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "report/stability_report.csv", index=False)
    print("Stability comparison SAVED!")


def exp_patchtst_configs(df):
    """Test different PatchTST configurations."""
    print("\n[EXP 5] PatchTST Configuration Analysis...")

    results = []
    patch_configs = [
        {'patch_len': 8, 'stride': 4, 'label': 'P8_S4'},
        {'patch_len': 16, 'stride': 8, 'label': 'P16_S8'},
        {'patch_len': 24, 'stride': 12, 'label': 'P24_S12'},
        {'patch_len': 32, 'stride': 16, 'label': 'P32_S16'},
    ]

    for config in patch_configs:
        print(f"   Testing PatchTST config: {config['label']}...")
        try:
            loaders, _, info = create_dataloaders_advanced(
                df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size']
            )

            cfg = CONFIG.copy()
            cfg['patch_len'] = config['patch_len']
            cfg['stride'] = config['stride']

            metrics = run_single_trial(
                loaders, info, "PatchTST",
                override_config=cfg,
                name_suffix=f"_{config['label']}"
            )
            metrics['Config'] = config['label']
            metrics['Patch_Len'] = config['patch_len']
            metrics['Stride'] = config['stride']
            results.append(metrics)

        except Exception as e:
            print(f"   Error: {e}")

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_DIR / "report/patchtst_configs.csv", index=False)
        print("PatchTST config analysis SAVED!")

    return results
