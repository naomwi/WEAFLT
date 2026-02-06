import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
from src.Utils.path import OUTPUT_DIR,DATA_DIR
from src.Model.Linear import DLinear,NLinear,LTSF_Linear
from src.Utils.parameter import CONFIG,device
from src.Utils.training import evaluate_model
from src.Data.data_loading import create_dataloaders_advanced
# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 150})
REPORT_DIR = OUTPUT_DIR/"report"
FIGURE_DIR = OUTPUT_DIR/"image"
os.makedirs(FIGURE_DIR, exist_ok=True)

def save_plot(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved plot: {path}")
    plt.close()

def plot_model_comparison():
    csv_path = OUTPUT_DIR/"comparison_results.csv"
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=df, x="Model", y="MAE", ax=ax1, palette="viridis")
    ax1.set_title("Comparison by MAE", fontsize=12, fontweight='bold')
    
    sns.barplot(data=df, x="Model", y="RMSE", ax=ax2, palette="magma")
    ax2.set_title("Comparison by RMSE", fontsize=12, fontweight='bold')
    
    for ax in [ax1, ax2]:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)
    save_plot("model_comparison.png")

def plot_runtime():
    csv_path = REPORT_DIR/"runtime_report.csv"
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Model", y="Time_s", palette="magma")
    
    plt.title("Inference Speed (Latency)", fontsize=14, fontweight='bold')
    plt.ylabel("Time per Sample (ms)")
    plt.xlabel("Model")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f ms', padding=3)
        
    save_plot("runtime_comparison.png")

def plot_horizon():
    csv_path = REPORT_DIR/"horizon_comparison.csv"
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

    if "MAE" not in df.columns or "RMSE" not in df.columns:
        print(f"Skipping plot_horizon: Missing MAE/RMSE columns. Columns: {df.columns.tolist()}")
        return

    plt.figure()
    sns.lineplot(data=df, x=horizon_col, y="MAE", marker="o", linewidth=2.5, label="MAE", color="blue")
    sns.lineplot(data=df, x=horizon_col, y="RMSE", marker="s", linewidth=2.5, label="RMSE", color="red")

    plt.title("Forecasting Performance vs Horizon", fontsize=14, fontweight='bold')
    plt.xlabel("Prediction Horizon (Hours)")
    plt.ylabel("Error")
    plt.xticks(df[horizon_col].unique())
    plt.legend()
    plt.grid(True, linestyle='--')

    save_plot("horizon_analysis.png")

def plot_alpha():
    csv_path = REPORT_DIR/"alpha_sensitivity.csv"
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    
    plt.figure()
    sns.lineplot(data=df, x="Alpha", y="MAE", marker="o", markersize=8, color="green", linewidth=2)
    
    # Đánh dấu điểm thấp nhất (Optimal)
    min_idx = df['MAE'].idxmin()
    best_alpha = df.loc[min_idx, "Alpha"]
    best_mae = df.loc[min_idx, "MAE"]
    plt.annotate(f'Optimal (α={best_alpha})', xy=(best_alpha, best_mae), xytext=(best_alpha, best_mae*1.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title("Impact of Event Weight (Alpha) on Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Alpha (Event Weight)")
    plt.ylabel("MAE Loss")
    plt.xticks(df["Alpha"].unique())
    
    save_plot("alpha_sensitivity.png")

def plot_ablation():
    csv_path = REPORT_DIR/"ablation_study.csv"
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(7, 6))
    ax = sns.barplot(data=df, x="Method", y="MAE", palette="rocket")
    
    plt.title("Ablation Study: Effectiveness of Proposed Method", fontsize=14, fontweight='bold')
    plt.ylabel("MAE (Lower is Better)")
    plt.xlabel("")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
        
    save_plot("ablation_study.png")

def plot_stability():
    csv_path = REPORT_DIR/"stability_report.csv"
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(6, 6))
    # Vẽ boxplot để thấy độ phân tán sai số qua các seed
    sns.boxplot(y=df["MAE"], width=0.4, color="skyblue")
    sns.stripplot(y=df["MAE"], color="black", size=8, jitter=True) # Vẽ các điểm seed thực tế
    
    mean_val = df["MAE"].mean()
    std_val = df["MAE"].std()
    
    plt.title(f"Model Stability (5 Random Seeds)\nMean: {mean_val:.4f} ± {std_val:.4f}", fontsize=14)
    plt.ylabel("MAE Distribution")
    
    save_plot("stability_analysis.png")

def plot_all_targets_sample(y_true, y_pred, target_names, sample_idx=0):
    
    num_targets = len(target_names)
    # Tạo lưới biểu đồ (3 hàng, 2 cột cho 6 targets)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
    axes = axes.flatten() # Trải phẳng để lặp dễ hơn

    for i in range(num_targets):
        true_series = y_true[sample_idx, :, i]
        pred_series = y_pred[sample_idx, :, i]
        
        ax = axes[i]
        ax.plot(true_series, label="Thực tế", color="#2ecc71", linewidth=2, marker='o', markersize=3)
        ax.plot(pred_series, label="Dự báo", color="#e74c3c", linestyle="--", linewidth=2, marker='x', markersize=3)
        
        ax.set_title(f"Thông số: {target_names[i]}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Bước dự báo (giờ)")
        ax.set_ylabel("Giá trị")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(f"all_targets_sample_{sample_idx}.png")
    

def plot_top_features_shap(shap_matrix, feature_names, top_n=15):
    """
    Trích xuất và vẽ Top N đặc trưng quan trọng nhất từ ma trận SHAP (2D: Time, Features)
    """
    # Tính trung bình cộng tác động của mỗi feature qua tất cả các bước thời gian
    feature_impact = np.mean(shap_matrix, axis=0) 
    
    # Tạo DataFrame để dễ sắp xếp
    df_feat = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_impact
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(8, 8))
    sns.barplot(data=df_feat, x="Importance", y="Feature", palette="Blues_d")
    
    plt.title(f"Top {top_n} Most Influential Features (Global SHAP)", fontsize=14, fontweight='bold')
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
    
    save_plot("top_features_importance.png")
def plot_all():
    print("\nGENERATING PLOTS...")
    plot_model_comparison()
    plot_runtime()
    plot_horizon()
    plot_alpha()
    plot_ablation()
    plot_stability()

    # Try to load processed data and generate sample plots
    try:
        # Try multiple possible paths
        possible_paths = [
            OUTPUT_DIR / "Final_Processed_Data.csv",
            DATA_DIR / "New_data/Training_data/Final_Processed_Data.csv",
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

        # Get dimensions from data
        input_dim = split_info['n_features']
        output_dim = split_info['n_targets']

        model = DLinear(input_dim=input_dim, output_dim=output_dim,
                       seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'])

        # Try to find a trained model
        model_paths = [
            OUTPUT_DIR / "model/best_model_DLinear_len24_alpha5.042.pth",
            OUTPUT_DIR / "model/best_model_DLinear_len24_alpha1.042.pth",
            OUTPUT_DIR / "model/best_model_DLinear_len24_alpha1.0None.pth",
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
        target_names = split_info.get('target_names', [])

        # Generate sample plots for valid indices
        max_samples = len(y_true)
        sample_indices = [i for i in [0, 10, 20, 50, 100] if i < max_samples]
        for i in sample_indices:
            plot_all_targets_sample(y_true, y_pred, target_names, sample_idx=i)

    except Exception as e:
        print(f"Error generating sample plots: {e}")

    print(f"All plots saved to '{FIGURE_DIR}' folder.")

if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR/"New_data/Training_data/Final_Processed_Data.csv")
    loaders,_,split_info = loaders, _, info = create_dataloaders_advanced(df, CONFIG['targets'], CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size'])
    train_loader, val_loader, test_loader = loaders
    model = DLinear(input_dim=87,output_dim=6,seq_len=CONFIG['seq_len'],pred_len=CONFIG['pred_len'])
    model_path = OUTPUT_DIR/"model/best_model_DLinear_len24_alpha1.0None.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    results = evaluate_model(model,test_loader,device,split_info)
    y_true = results['raw_data']['actuals'] # (N, 24, 6)
    y_pred = results['raw_data']['preds']   # (N, 24, 6)
    target_names = split_info.get('target_names', [])
    
    for i in [0,10,20,50,100]:
        plot_all_targets_sample(y_true, y_pred, target_names, sample_idx=i)

