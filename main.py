import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from src.Utils.parameter import CONFIG,RUNTIME_LOG,device
from src.seed import seed_everything
from src.Utils.support_class import EarlyStopping
from src.Data.data_loading import load_data_source_separate
from src.CEEMD.ceemd_filter import apply_ceemd_decomposition
from src.Utils.path import OUTPUT_DIR,DATA_DIR
from src.Experiments.deploy_experiments import exp_ablation_study,exp_alpha_sensitivity,exp_horizon_comparison,exp_model_comparison,exp_stability
from src.Experiments.visual import plot_all
# --- config ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


# --- preprocess data ---
def data_processing():

    df = load_data_source_separate()

    try:
        cols_to_decomposed =  ['Flow_log', 'Turbidity_log', 'EC_log'] 
        df_decomposed, info = apply_ceemd_decomposition(pd.read_csv(DATA_DIR/'New_data/Training_data/input.csv'), cols_to_decomposed, n_imfs=4, trials=10)
        print("Stats IMF:", info)
        print("New Columns", [c for c in df_decomposed.columns if "IMF" in c])
        df_decomposed.to_csv(DATA_DIR/"New_data/Training_data/Final_Dataset_Multivariate_CEEMDAN.csv", index=False)
    except Exception as e:
        print(f"CEEMD skipped: {e}")
        exit(0)
    return df_decomposed

def main():
    print("="*80)
    print("WATER QUALITY FORECASTING")
    print("="*80)
    
    # Seed for reproducibility
    seed_everything(42)
    
    #df_main = data_processing()
    df_main = pd.read_csv(CONFIG['csv_path'])
    print("RUNNING EXPERIMENT")
    print("="*80)
    # Chạy lần lượt (Bạn có thể comment dòng nào không thích chạy)
    print("--> MODEL COMPARISION <-- ")
    exp_model_comparison(df_main)
    print("="*80)

    print("--> ALPHA SENSITIVITY <-- ")
    exp_alpha_sensitivity(df_main)
    print("="*80)

    print("--> HORIZON COMPARISON <-- ")
    exp_stability(df_main)
    print("="*80)

    print("--> STABILITY <-- ")
    exp_ablation_study(df_main)
    print("="*80)
    
    # Riêng cái này chạy lâu nhất vì phải tạo lại data liên tục
    # exp_horizon_comparison(df_main)

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
