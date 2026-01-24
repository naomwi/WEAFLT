import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.Model.TimeSHAP import temporal_attribution_enhanced
from src.Utils.parameter import RUNTIME_LOG,STABILITY_LOG
from src.seed import seed_everything

def run_stability_analysis(model, dataset, test_loader, device, model_name="Model"):
    """
    🔧 FIXED: 
    1. Attribution Variance now measures variance of attribution values, not correlations
    2. Renamed "Seed Consistency" to "Perturbation Consistency"
    3. TimeSHAP-compliant baseline (mean across all features)

    🛡️ COMPREHENSIVE STABILITY ANALYSIS
    """
    print(f"\n🛡️ [8/8] Running Stability Analysis for {model_name}...")

    # Select multiple samples
    test_indices = list(test_loader.dataset.indices)
    selected_indices = []

    # Try to find event samples
    for idx in test_indices:
        _, _, evt = dataset[idx]
        if evt.sum() > 0:
            selected_indices.append(idx)
        if len(selected_indices) >= 5:
            break

    # Fill with non-event samples if needed
    while len(selected_indices) < 10 and len(selected_indices) < len(test_indices):
        idx = test_indices[len(selected_indices)]
        if idx not in selected_indices:
            selected_indices.append(idx)

    # Results containers
    noise_correlations = []
    perturbation_jaccards = []
    attr_variances = []

    for sample_idx, idx in enumerate(selected_indices):
        x, y, evt = dataset[idx]
        x_np = x.numpy()

        # 🔧 FIXED: TimeSHAP-compliant baseline (mean across ALL features)
        baseline = np.tile(np.mean(x_np, axis=0, keepdims=True), (x_np.shape[0], 1))

        # 1. Baseline Attribution
        # ⏱️ MEASURE XAI TIME
        xai_start = time.time()
        base_attr = temporal_attribution_enhanced(model, dataset, x_np, baseline, 4, device)
        xai_time = (time.time() - xai_start) * 1000

        if sample_idx == 0:
            RUNTIME_LOG.append({
                "Stage": "XAI",
                "Model": model_name,
                "Time_s": xai_time,
                "Unit": "ms/sample"
            })

        # 2. Noise Robustness (5 runs with 5% noise)
        sample_correlations = []
        noisy_attributions = []

        for _ in range(5):
            noise = np.random.normal(0, 0.05, x_np.shape)
            noisy_x = x_np + noise
            # Recompute baseline for noisy input
            noisy_baseline = np.tile(np.mean(noisy_x, axis=0, keepdims=True), (noisy_x.shape[0], 1))
            noisy_attr = temporal_attribution_enhanced(model, dataset, noisy_x, noisy_baseline, 4, device)

            noisy_attributions.append(noisy_attr)

            corr, _ = spearmanr(base_attr, noisy_attr)
            sample_correlations.append(corr if not np.isnan(corr) else 0.0)

        noise_correlations.extend(sample_correlations)

        # 🔧 FIXED: Attribution Variance (variance of attribution values, not correlations)
        noisy_attributions = np.stack(noisy_attributions, axis=0)  # [K=5, T_segments]
        attr_var = np.mean(np.var(noisy_attributions, axis=0))  # Mean variance across time segments
        attr_variances.append(attr_var)

        # 3. 🔧 FIXED: Renamed to "Perturbation Consistency"
        top_k = min(3, len(base_attr))
        base_top = set(np.argsort(base_attr)[-top_k:])

        sample_jaccards = []
        for seed in [1, 2, 3, 4, 5]:
            seed_everything(seed)

            # Micro perturbation to test consistency
            micro_noise = np.random.normal(0, 0.01, x_np.shape)
            pert_x = x_np + micro_noise
            pert_baseline = np.tile(np.mean(pert_x, axis=0, keepdims=True), (pert_x.shape[0], 1))
            pert_attr = temporal_attribution_enhanced(model, dataset, pert_x, pert_baseline, 4, device)

            pert_top = set(np.argsort(pert_attr)[-top_k:])

            intersection = len(base_top.intersection(pert_top))
            union = len(base_top.union(pert_top))
            jaccard = intersection / union if union > 0 else 0.0
            sample_jaccards.append(jaccard)

        perturbation_jaccards.extend(sample_jaccards)

    # Compute summary statistics
    mean_noise_corr = np.mean(noise_correlations)
    mean_jaccard = np.mean(perturbation_jaccards)
    mean_attr_var = np.mean(attr_variances)

    print(f"   ✅ Noise Robustness (Spearman): {mean_noise_corr:.4f}")
    print(f"   ✅ Perturbation Consistency (Jaccard): {mean_jaccard:.4f}")
    print(f"   ✅ Attribution Variance: {mean_attr_var:.6f}")

    STABILITY_LOG.append({
        "Model": model_name,
        "Noise_Robustness_Spearman": mean_noise_corr,
        "Perturbation_Consistency_Jaccard": mean_jaccard,
        "Attribution_Variance": mean_attr_var,
        "N_Samples": len(selected_indices)
    })

    return {
        "noise_correlations": noise_correlations,
        "perturbation_jaccards": perturbation_jaccards,
        "attr_variances": attr_variances
    }