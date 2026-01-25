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
from src.Experiments.deploy_experiments import exp_ablation_study,exp_alpha_sensitivity,exp_horizon_comparison,exp_model_comparison,exp_stability,exp_xai_analysis
from src.Experiments.visual import plot_all
# --- config ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


# --- preprocess data ---
def data_processing():
    df = pd.read_csv(DATA_DIR/'New_data/USGS/water_data_2021_2025_clean.csv')
    processor = DataProcessor(CONFIG)
    df_final = processor.run_pipeline(df)
    cols = df_final.columns
    print(f"Total columns: {len(cols)}")
    print("Sample Columns:", cols[:5])
    print("Sample Residue:", [c for c in cols if 'residue' in c][:2])
    print("Sample Event:", [c for c in cols if 'event_flag' in c][:2])
    print(f"Total columns: {len(cols)}")
    print("Sample Columns:", cols[:5])
    print("Sample Residue:", [c for c in cols if 'residue' in c][:2])
    print("Sample Event:", [c for c in cols if 'event_flag' in c][:2])
    df_final.to_csv(DATA_DIR/"New_data/Training_data/Final_Processed_Data.csv", index=False)
def main():
    print("="*80)
    print("WATER QUALITY FORECASTING")
    print("="*80)
    
    seed_everything(42)
    
    df_main = data_processing()
    
    print("RUNNING EXPERIMENT")
    print("="*80)
    # Chạy lần lượt (Bạn có thể comment dòng nào không thích chạy)
    print("--> MODEL COMPARISION <-- ")
    exp_model_comparison(df_main)
    print("="*80)

    print("--> ALPHA SENSITIVITY <-- ")
    exp_alpha_sensitivity(df_main)
    print("="*80)

    print("--> STABILITY COMPARISON <-- ")
    exp_stability(df_main)
    print("="*80)

    print("--> ABLATION <-- ")
    exp_ablation_study(df_main)
    print("="*80)
    
    # Riêng cái này chạy lâu nhất vì phải tạo lại data liên tục
    exp_horizon_comparison(df_main)
    exp_xai_analysis(df_main)

    print("Save RUN_TIME_LOG")
    df_runtime = pd.DataFrame(RUNTIME_LOG)
    df_runtime.to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)
    
    print("Ploting report")
    print("="*80)
    plot_all()
    
    print("FINSH EXPERIMENT!!!")
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    os.makedirs(OUTPUT_DIR/'report',exist_ok=True)
    os.makedirs(OUTPUT_DIR/'image',exist_ok=True)
    os.makedirs(OUTPUT_DIR/'model',exist_ok=True)
    main()
