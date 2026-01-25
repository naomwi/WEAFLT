import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.Utils.path import OUTPUT_DIR

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
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    
    plt.figure()
    sns.lineplot(data=df, x="Horizon", y="MAE", marker="o", linewidth=2.5, label="MAE", color="blue")
    sns.lineplot(data=df, x="Horizon", y="RMSE", marker="s", linewidth=2.5, label="RMSE", color="red")
    
    plt.title("Forecasting Performance vs Horizon", fontsize=14, fontweight='bold')
    plt.xlabel("Prediction Horizon (Hours)")
    plt.ylabel("Error")
    plt.xticks(df["Horizon"].unique()) # Đảm bảo hiện đúng các mốc 24, 48...
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

def plot_all():
    print("\nGENERATING PLOTS...")
    plot_model_comparison()
    plot_runtime()
    plot_horizon()
    plot_alpha()
    plot_ablation()
    plot_stability()
    print(f"All plots saved to '{FIGURE_DIR}' folder.")
