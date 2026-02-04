import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.Utils.path import OUTPUT_DIR

# Optional SHAP import with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap package not installed. XAI features will be limited.")
    print("Install with: pip install shap")

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
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP package is required for XAI analysis. "
                "Install with: pip install shap"
            )

        self.device = device
        self.model = model.eval()
        self.explainer = None

        print(f"XAI: Initializing background data...")

        # Collect background data
        raw_data = []
        count = 0
        limit = 500
        for x, _, _, _ in train_loader:
            raw_data.append(x)
            count += x.shape[0]
            if count >= limit:
                break

        data_pool = torch.cat(raw_data)[:limit]

        # K-Means Background selection
        from sklearn.cluster import KMeans
        flat_data = data_pool.reshape(data_pool.shape[0], -1).cpu().numpy()
        kmeans = KMeans(n_clusters=num_background, n_init=10).fit(flat_data)

        centroids = kmeans.cluster_centers_.reshape(
            num_background, data_pool.shape[1], data_pool.shape[2]
        )
        self.background = torch.FloatTensor(centroids).to(device)

        self.wrapper = ShapModelWrapper(self.model).to(device)

        try:
            self.explainer = shap.DeepExplainer(self.wrapper, self.background)
            print("XAI: DeepExplainer initialized successfully.")
        except Exception as e:
            print(f"XAI: DeepExplainer failed ({e}). Using GradientExplainer fallback.")
            try:
                self.explainer = shap.GradientExplainer(self.wrapper, self.background)
                print("XAI: GradientExplainer initialized as fallback.")
            except Exception as e2:
                print(f"XAI: All explainers failed ({e2}).")

    def explain_sample(self, input_sample, feature_names=None):
        """
        Explain a single data sample using SHAP values.
        """
        if self.explainer is None:
            raise RuntimeError("XAI explainer not initialized. Check for errors during __init__.")

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

    def compute_feature_importance(
        self,
        data_loader,
        n_samples: int = 100,
        feature_names=None
    ):
        """
        Compute feature importance scores across multiple samples.

        Args:
            data_loader: DataLoader to sample from
            n_samples: Number of samples to analyze
            feature_names: List of feature names

        Returns:
            Dictionary with feature importance scores
        """
        if self.explainer is None:
            raise RuntimeError("XAI explainer not initialized.")

        all_shap = []
        count = 0

        for x, _, _, _ in data_loader:
            if count >= n_samples:
                break

            batch_size = min(x.shape[0], n_samples - count)
            x_batch = x[:batch_size].to(self.device).requires_grad_()

            try:
                shap_values = self.explainer.shap_values(x_batch)

                if isinstance(shap_values, list):
                    shap_array = np.mean([np.abs(s) for s in shap_values], axis=0)
                else:
                    shap_array = np.abs(shap_values)

                all_shap.append(shap_array)
                count += batch_size

            except Exception as e:
                print(f"Warning: SHAP computation failed for batch: {e}")
                continue

        if not all_shap:
            return {}

        # Aggregate SHAP values
        combined_shap = np.concatenate(all_shap, axis=0)

        # Mean absolute SHAP per feature (averaged over time and samples)
        if combined_shap.ndim == 3:
            # [samples, time, features]
            feature_importance = np.mean(combined_shap, axis=(0, 1))
        else:
            feature_importance = np.mean(combined_shap, axis=0)

        # Normalize to percentages
        total = feature_importance.sum()
        if total > 0:
            feature_importance_pct = feature_importance / total * 100
        else:
            feature_importance_pct = feature_importance

        # Create result dictionary
        if feature_names and len(feature_names) == len(feature_importance):
            result = {name: float(imp) for name, imp in zip(feature_names, feature_importance_pct)}
        else:
            result = {f"Feature_{i}": float(imp) for i, imp in enumerate(feature_importance_pct)}

        return result

    def get_imf_contributions(self, feature_importance: dict):
        """
        Extract IMF-specific contributions from feature importance.

        Args:
            feature_importance: Dictionary of feature -> importance

        Returns:
            Dictionary with IMF contribution analysis
        """
        import re

        imf_contributions = {}
        non_imf_contributions = {}

        imf_pattern = re.compile(r'(.+)_(IMF_\d+|residue)$')

        for feat, importance in feature_importance.items():
            match = imf_pattern.match(feat)
            if match:
                var_name = match.group(1)
                comp_type = match.group(2)

                if var_name not in imf_contributions:
                    imf_contributions[var_name] = {}

                imf_contributions[var_name][comp_type] = importance
            else:
                non_imf_contributions[feat] = importance

        # Calculate totals per variable
        summary = {}
        for var_name, components in imf_contributions.items():
            total = sum(components.values())
            summary[var_name] = {
                'total_contribution': total,
                'components': components,
                'n_components': len(components)
            }

        return {
            'imf_contributions': imf_contributions,
            'non_imf_contributions': non_imf_contributions,
            'summary': summary
        }

    def suggest_pruning(
        self,
        feature_importance: dict,
        threshold: float = 95.0,
        min_features: int = 10
    ):
        """
        Suggest features to prune based on SHAP importance.

        Args:
            feature_importance: Dictionary of feature -> importance (%)
            threshold: Cumulative importance threshold to keep
            min_features: Minimum features to keep

        Returns:
            Dictionary with pruning suggestions
        """
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Accumulate until threshold
        cumulative = 0
        features_to_keep = []

        for feat, importance in sorted_features:
            if cumulative < threshold or len(features_to_keep) < min_features:
                features_to_keep.append(feat)
                cumulative += importance

        features_to_prune = [f for f, _ in sorted_features if f not in features_to_keep]

        return {
            'features_to_keep': features_to_keep,
            'features_to_prune': features_to_prune,
            'n_original': len(sorted_features),
            'n_kept': len(features_to_keep),
            'n_pruned': len(features_to_prune),
            'reduction_pct': len(features_to_prune) / len(sorted_features) * 100 if sorted_features else 0,
            'cumulative_importance_kept': cumulative
        }

    def plot_feature_importance(
        self,
        feature_importance: dict,
        top_k: int = 30,
        save_path=None
    ):
        """
        Plot top-K feature importance bar chart.

        Args:
            feature_importance: Dictionary of feature -> importance
            top_k: Number of top features to show
            save_path: Path to save figure
        """
        # Sort and get top K
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance (%)')
        plt.ylabel('Feature')
        plt.title(f'Top {top_k} Feature Importance (SHAP-based)')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        else:
            save_path = OUTPUT_DIR / "image/feature_importance.png"
            os.makedirs(save_path.parent, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()