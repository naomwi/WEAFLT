import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.Utils.path import OUTPUT_DIR

# --- 1. CLASS WRAPPER ---
class ShapModelWrapper(nn.Module):
    """
    Biến đổi đầu ra 3D [Batch, Time, Feat] thành 2D [Batch, Time*Feat]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        return out.reshape(out.shape[0], -1)

# --- 2. XAI HANDLER ---
class XAI_Handler:
    def __init__(self, model, train_loader, device, num_background=20):
        self.device = device
        self.model = model.eval()
        
        print(f"XAI: Initializing background data...")
        
        # Lấy dữ liệu nền
        raw_data = []
        count = 0
        limit = 500 
        for x, _, _, _ in train_loader:
            raw_data.append(x)
            count += x.shape[0]
            if count >= limit: break
            
        data_pool = torch.cat(raw_data)[:limit]
        
        # K-Means Background
        from sklearn.cluster import KMeans
        flat_data = data_pool.reshape(data_pool.shape[0], -1).cpu().numpy()
        kmeans = KMeans(n_clusters=num_background, n_init=10).fit(flat_data)
        
        centroids = kmeans.cluster_centers_.reshape(num_background, data_pool.shape[1], data_pool.shape[2])
        self.background = torch.FloatTensor(centroids).to(device)
        
        self.wrapper = ShapModelWrapper(self.model).to(device)
        
        try:
            self.explainer = shap.DeepExplainer(self.wrapper, self.background)
            print("XAI: DeepExplainer initialized successfully.")
        except Exception as e:
            print(f"XAI: DeepExplainer failed ({e}).")

    def explain_sample(self, input_sample, feature_names=None):
        """
        Giải thích 1 mẫu dữ liệu cụ thể.
        """
        if input_sample.dim() == 2:
            input_sample = input_sample.unsqueeze(0)
            
        input_sample = input_sample.to(self.device).requires_grad_()
        
        shap_values = self.explainer.shap_values(input_sample)
        
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values) 
            shap_matrix = np.sum(np.abs(shap_array), axis=0)
            
        elif shap_values.ndim == 4:
            shap_matrix = np.sum(np.abs(shap_values), axis=-1)
            
        else:
            shap_matrix = np.abs(shap_values)

        if shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[0] 
      

        self.plot_shap_heatmap(shap_matrix, feature_names)
        
        return shap_matrix

    def plot_shap_heatmap(self, shap_matrix, feature_names):
        """
        Vẽ Heatmap: Trục tung là Features, Trục hoành là Time Lag.
        shap_matrix Input: [Time, Features] (VD: 96, 87)
        """
        
        shap_matrix_T = shap_matrix.T 
        
        plt.figure(figsize=(12, 12)) 
        
        ax = sns.heatmap(
            shap_matrix_T, 
            cmap="viridis", 
            xticklabels=12, 
            yticklabels=feature_names if feature_names else "auto",
            cbar_kws={'label': 'Cumulative Feature Impact'}
        )
        
        plt.title("Feature Importance over Time (SHAP Heatmap)")
        plt.xlabel("Time Lag (Past Steps)")
        plt.ylabel("Input Features")
        
        if feature_names and len(feature_names) > 50:
            ax.tick_params(axis='y', labelsize=6)
        
        save_path = OUTPUT_DIR / "image/xai_heatmap.png"
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
  
        plt.close()
        print(f"XAI Heatmap saved to: {save_path}")