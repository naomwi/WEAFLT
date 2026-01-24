import os
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random

from src.Data.data_loading import create_dataloaders_advanced
from src.Model.Linear import LTSF_Linear,DLinear,NLinear
from src.Utils.parameter import CONFIG,device
from src.Utils.path import OUTPUT_DIR
from src.Utils.support_class import EventWeightedMSE
from src.Utils.training import evaluate_model,train_model
from src.seed import seed_everything

def get_model(model_name, input_dim, output_dim, seq_len, pred_len, device):
    """Factory để tạo model nhanh"""
    if model_name == "NLinear": 
        model = NLinear(input_dim, output_dim, seq_len, pred_len)
    elif model_name == "DLinear": 
        model = DLinear(input_dim, output_dim, seq_len, pred_len)
    else: 
        model = LTSF_Linear(input_dim, output_dim, seq_len, pred_len)
    return model.to(device)


def run_single_trial(loaders, split_info, model_name, override_config=None):
    """
    Hàm chạy 1 lượt Train-Eval trọn vẹn.
    Nhận loaders đã load sẵn để tiết kiệm thời gian.
    """
    cfg = CONFIG.copy()
    if override_config: cfg.update(override_config)
    
    train_loader, val_loader, test_loader = loaders
    in_dim, out_dim = split_info['n_features'], split_info['n_targets']
    
    model = get_model(model_name, in_dim, out_dim, cfg['seq_len'], cfg['pred_len'], device)
    
    criterion = EventWeightedMSE(alpha=cfg['event_weight']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    
    # Lưu tên model kèm config để tránh ghi đè file trọng số
    unique_name = f"{model_name}_len{cfg['pred_len']}_alpha{cfg['event_weight']}"
    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                        cfg['epochs'], device, model_name=unique_name)
    
    # Measure Inference Time (GPU Synchronized)
    model.eval()
    times = []
    with torch.no_grad():
        for x, _, _, _ in test_loader:
            x = x.to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            st = time.time()
            _ = model(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            times.append((time.time()-st)/x.size(0))
    
    metrics = evaluate_model(model, test_loader, device, split_info)
    metrics['Inference_ms'] = np.mean(times) * 1000
    
    return metrics

# 1. MODEL COMPARISON (So sánh Model)
def exp_model_comparison(df):
    print("\n🧪 EXP 1: Model Comparison & Runtime...")
    
    loaders, _, info = create_dataloaders_advanced(df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size'])
    
    results = []
    for m in ["DLinear", "NLinear", "LTSF_Linear"]:
        print(f"   Running {m}...")
        metrics = run_single_trial(loaders, info, m)
        metrics['Model'] = m
        results.append(metrics)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{OUTPUT_DIR}/comparison_results.csv", index=False)
    
    # Tách file Runtime riêng
    df_res[['Model', 'Inference_ms']].to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)
    print("Model Comparision report SAVED!!")

# 2. EXP: ALPHA SENSITIVITY (Độ nhạy Loss)
def exp_alpha_sensitivity(df):
    print("\n🧪 EXP 2: Alpha Sensitivity...")
    
    loaders, _, info = create_dataloaders_advanced(df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size'])
    
    results = []
    alphas = [1.0, 3.0, 5.0, 10.0]
    
    for alpha in alphas:
        print(f"   Alpha = {alpha}...")
        metrics = run_single_trial(loaders, info, "DLinear", override_config={'event_weight': alpha})
        metrics['Alpha'] = alpha
        results.append(metrics)
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)
    print("Alpha sensitivity comparision SAVED!!")

# 3. EXP: HORIZON COMPARISON (Dự báo xa)
def exp_horizon_comparison(df):
    print("\n🧪 EXP 3: Forecasting Horizon...")
    
    results = []
    horizons = [24, 48, 96, 192]
    
    for h in horizons:
        print(f"   Horizon = {h}...")
        loaders, _, info = create_dataloaders_advanced(
            df, CONFIG['targets'], CONFIG['seq_len'], 
            pred_len=h,
            batch_size=CONFIG['batch_size']
        )
        
        metrics = run_single_trial(loaders, info, "DLinear", override_config={'pred_len': h})
        metrics['Horizon'] = h
        results.append(metrics)
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)
    print("Horizon comparision comparision SAVED!!")

# 4. EXP: STABILITY (Kiểm tra độ ổn định)  
def exp_stability(df):
    
    loaders, _, info = create_dataloaders_advanced(df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size'])
    
    results = []
    seeds = [42, 100, 2024, 777, 99]
    
    for s in seeds:
        print(f"   Seed = {s}...")
        seed_everything(s) # Đặt seed trước khi init model
        metrics = run_single_trial(loaders, info, "DLinear")
        metrics['Seed'] = s
        results.append(metrics)
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)
    print("Stability comparision SAVED!!")

# 5. EXP: ABLATION STUDY (Nghiên cứu cắt bỏ)
def exp_ablation_study(df):
    print("\n🧪 EXP 5: Ablation Study...")
    
    loaders, _, info = create_dataloaders_advanced(df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size'])
    
    results = []
    
    print("   Running Proposed Method...")
    m1 = run_single_trial(loaders, info, "DLinear", override_config={'event_weight': 5.0})
    m1['Method'] = "Proposed (Weighted Loss)"
    results.append(m1)
    
    print("   Running Baseline Method...")
    m2 = run_single_trial(loaders, info, "DLinear", override_config={'event_weight': 1.0})
    m2['Method'] = "Baseline (Standard MSE)"
    results.append(m2)
    
    pd.DataFrame(results).to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)
    print("Ablation study SAVED!!")