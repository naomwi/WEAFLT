"""
TimeSHAP Analysis with Pruning
Explain feature/timestep importance with computational efficiency
Includes runtime measurement and stability analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Callable, Optional, Tuple
from pathlib import Path
import warnings


def run_timeshap_analysis(
    model_fn: Callable,
    X_sample: np.ndarray,
    feature_names: List[str],
    n_background: int = 100,
    top_k_timesteps: int = 20,
    output_dir: Path = None
) -> Dict:
    """
    Run TimeSHAP analysis with pruning for efficiency.

    Args:
        model_fn: Function that takes input and returns prediction
        X_sample: Sample to explain (seq_len, n_features)
        feature_names: Names of features
        n_background: Number of background samples for SHAP
        top_k_timesteps: Focus on top-k most important timesteps (pruning)
        output_dir: Directory to save results

    Returns:
        Dictionary with SHAP analysis results
    """
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP not installed. Install with: pip install shap")
        return _simple_importance_analysis(model_fn, X_sample, feature_names, top_k_timesteps)

    seq_len, n_features = X_sample.shape

    # Phase 1: Quick importance estimation (pruning)
    print("Phase 1: Estimating timestep importance for pruning...")
    timestep_importance = _estimate_timestep_importance(model_fn, X_sample)

    # Select top-k timesteps
    top_indices = np.argsort(timestep_importance)[-top_k_timesteps:]
    print(f"  Focusing on top {top_k_timesteps} timesteps: {sorted(top_indices)[:5]}...")

    # Phase 2: Detailed SHAP on pruned timesteps
    print("Phase 2: Running SHAP on pruned timesteps...")

    # Create pruned input
    X_pruned = X_sample[top_indices, :]  # (top_k, n_features)

    # Simple perturbation-based importance for pruned features
    feature_importance = np.zeros((top_k_timesteps, n_features))

    for t in range(top_k_timesteps):
        for f in range(n_features):
            # Perturb feature f at timestep t
            X_pert = X_sample.copy()
            X_pert[top_indices[t], f] = 0  # Zero out

            # Measure impact
            orig_pred = model_fn(X_sample.reshape(1, seq_len, n_features))
            pert_pred = model_fn(X_pert.reshape(1, seq_len, n_features))

            importance = np.abs(orig_pred - pert_pred).mean()
            feature_importance[t, f] = importance

    # Aggregate results
    timestep_importance_pruned = feature_importance.sum(axis=1)
    feature_importance_total = feature_importance.sum(axis=0)

    results = {
        'top_timesteps': top_indices,
        'timestep_importance': timestep_importance_pruned,
        'feature_importance': feature_importance_total,
        'feature_names': feature_names,
        'detailed_importance': feature_importance,
    }

    # Create summary
    summary = []
    for i, idx in enumerate(top_indices):
        for j, fname in enumerate(feature_names):
            summary.append({
                'Timestep': idx,
                'Feature': fname,
                'Importance': feature_importance[i, j]
            })

    results['summary_df'] = pd.DataFrame(summary)

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results['summary_df'].to_csv(output_dir / 'timeshap_analysis.csv', index=False)

        # Feature importance summary
        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance_total
        }).sort_values('Importance', ascending=False)
        feat_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    return results


def _estimate_timestep_importance(model_fn, X_sample: np.ndarray) -> np.ndarray:
    """
    Quick estimation of timestep importance using gradient approximation.
    """
    seq_len, n_features = X_sample.shape
    importance = np.zeros(seq_len)

    base_pred = model_fn(X_sample.reshape(1, seq_len, n_features))

    for t in range(seq_len):
        X_pert = X_sample.copy()
        X_pert[t, :] = 0  # Zero out entire timestep

        pert_pred = model_fn(X_pert.reshape(1, seq_len, n_features))
        importance[t] = np.abs(base_pred - pert_pred).mean()

    return importance


def _simple_importance_analysis(
    model_fn: Callable,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_k: int
) -> Dict:
    """
    Simple perturbation-based importance analysis (fallback when SHAP unavailable).
    """
    seq_len, n_features = X_sample.shape

    # Timestep importance
    timestep_importance = _estimate_timestep_importance(model_fn, X_sample)

    # Feature importance (average over timesteps)
    feature_importance = np.zeros(n_features)
    base_pred = model_fn(X_sample.reshape(1, seq_len, n_features))

    for f in range(n_features):
        X_pert = X_sample.copy()
        X_pert[:, f] = 0

        pert_pred = model_fn(X_pert.reshape(1, seq_len, n_features))
        feature_importance[f] = np.abs(base_pred - pert_pred).mean()

    return {
        'timestep_importance': timestep_importance,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'top_timesteps': np.argsort(timestep_importance)[-top_k:],
    }


# =============================================================================
# RUNTIME MEASUREMENT
# =============================================================================

def measure_shap_runtime(
    model_fn: Callable,
    X_data: np.ndarray,
    feature_names: List[str],
    pruning_ratios: List[float] = [0.0, 0.25, 0.5, 0.75],
    n_samples: int = 10,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Measure SHAP computation runtime with different pruning ratios.

    Args:
        model_fn: Function that takes input and returns prediction
        X_data: Data samples (n_samples, seq_len, n_features)
        feature_names: Names of features
        pruning_ratios: List of pruning ratios (0.0 = no pruning, 0.75 = keep 25%)
        n_samples: Number of samples to test
        output_dir: Directory to save results

    Returns:
        DataFrame with runtime measurements
    """
    if X_data.ndim == 2:
        X_data = X_data.reshape(1, *X_data.shape)

    seq_len = X_data.shape[1]
    results = []

    print("\n" + "=" * 60)
    print("RUNTIME MEASUREMENT")
    print("=" * 60)

    # Baseline: full computation (0% pruning)
    base_runtime = None

    for ratio in pruning_ratios:
        top_k = max(1, int(seq_len * (1 - ratio)))
        print(f"\nPruning ratio: {ratio*100:.0f}% (keeping {top_k} of {seq_len} timesteps)")

        runtimes = []
        for i in range(min(n_samples, X_data.shape[0])):
            X_sample = X_data[i] if X_data.ndim == 3 else X_data

            start = time.time()
            _ = run_timeshap_analysis(
                model_fn, X_sample, feature_names,
                top_k_timesteps=top_k
            )
            elapsed = time.time() - start
            runtimes.append(elapsed)

        avg_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)

        if ratio == 0.0:
            base_runtime = avg_runtime

        speedup = base_runtime / avg_runtime if avg_runtime > 0 else 1.0

        results.append({
            'pruning_ratio': ratio,
            'top_k_timesteps': top_k,
            'runtime_mean': avg_runtime,
            'runtime_std': std_runtime,
            'speedup': speedup,
        })

        print(f"  Runtime: {avg_runtime:.2f}s ± {std_runtime:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

    df = pd.DataFrame(results)

    # Save and plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        df.to_csv(output_dir / 'runtime_comparison.csv', index=False)

        # Plot
        _plot_runtime_comparison(df, output_dir)

    return df


def _plot_runtime_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot runtime comparison chart."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    # Runtime bars
    bars = ax1.bar(x - width/2, df['runtime_mean'], width,
                   yerr=df['runtime_std'], capsize=5,
                   label='Runtime', color='steelblue', alpha=0.8)
    ax1.set_xlabel('Pruning Ratio (%)')
    ax1.set_ylabel('Runtime (seconds)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r*100:.0f}%" for r in df['pruning_ratio']])

    # Speedup line on secondary axis
    ax2 = ax1.twinx()
    line = ax2.plot(x, df['speedup'], 'o-', color='darkorange',
                    linewidth=2, markersize=8, label='Speedup')
    ax2.set_ylabel('Speedup (x)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Add value labels
    for i, (rt, sp) in enumerate(zip(df['runtime_mean'], df['speedup'])):
        ax1.text(i - width/2, rt + df['runtime_std'].iloc[i] + 0.1,
                 f'{rt:.1f}s', ha='center', va='bottom', fontsize=9)
        ax2.text(i, sp + 0.1, f'{sp:.1f}x', ha='center', va='bottom',
                 fontsize=9, color='darkorange')

    plt.title('SHAP Runtime vs Pruning Ratio', fontsize=14, fontweight='bold')
    fig.tight_layout()

    plt.savefig(output_dir / 'runtime_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'runtime_comparison.png'}")


# =============================================================================
# STABILITY ANALYSIS
# =============================================================================

def analyze_shap_stability(
    model_fn: Callable,
    X_sample: np.ndarray,
    feature_names: List[str],
    n_runs: int = 10,
    top_k_timesteps: int = 20,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Analyze stability of SHAP values across multiple runs.

    Args:
        model_fn: Function that takes input and returns prediction
        X_sample: Sample to explain (seq_len, n_features)
        feature_names: Names of features
        n_runs: Number of runs for stability analysis
        top_k_timesteps: Number of timesteps to focus on
        output_dir: Directory to save results

    Returns:
        Dictionary with stability metrics
    """
    print("\n" + "=" * 60)
    print(f"STABILITY ANALYSIS (n_runs={n_runs})")
    print("=" * 60)

    all_feature_importance = []
    all_top_timesteps = []

    for run in range(n_runs):
        # Add small noise to simulate stochasticity
        np.random.seed(run)
        X_noisy = X_sample + np.random.randn(*X_sample.shape) * 1e-6

        results = _simple_importance_analysis(
            model_fn, X_noisy, feature_names, top_k_timesteps
        )

        all_feature_importance.append(results['feature_importance'])
        all_top_timesteps.append(set(results['top_timesteps']))

        print(f"  Run {run+1}/{n_runs} completed")

    # Convert to arrays
    importance_array = np.array(all_feature_importance)  # (n_runs, n_features)

    # Calculate statistics
    mean_importance = importance_array.mean(axis=0)
    std_importance = importance_array.std(axis=0)
    cv_importance = std_importance / (mean_importance + 1e-8)  # Coefficient of variation

    # Top-k consistency: how often do the same timesteps appear in top-k?
    all_sets = all_top_timesteps
    consistency_scores = []
    for i in range(len(all_sets)):
        for j in range(i+1, len(all_sets)):
            intersection = len(all_sets[i] & all_sets[j])
            union = len(all_sets[i] | all_sets[j])
            consistency_scores.append(intersection / union if union > 0 else 0)

    top_k_consistency = np.mean(consistency_scores) * 100

    stability_results = {
        'mean_importance': mean_importance,
        'std_importance': std_importance,
        'cv_importance': cv_importance,
        'top_k_consistency': top_k_consistency,
        'feature_names': feature_names,
        'n_runs': n_runs,
    }

    # Print summary
    print(f"\nStability Summary:")
    print(f"  Top-K Consistency: {top_k_consistency:.1f}%")
    print(f"\nFeature Stability (CV = lower is better):")
    for fname, cv in zip(feature_names, cv_importance):
        print(f"  {fname}: CV = {cv:.4f}")

    # Save and plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        stability_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Importance': mean_importance,
            'Std_Importance': std_importance,
            'CV': cv_importance,
        })
        stability_df.to_csv(output_dir / 'stability_analysis.csv', index=False)

        # Plot
        _plot_stability_analysis(stability_results, output_dir)

    return stability_results


def _plot_stability_analysis(results: Dict, output_dir: Path) -> None:
    """Plot stability analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    feature_names = results['feature_names']
    mean_imp = results['mean_importance']
    std_imp = results['std_importance']

    # Error bar plot
    x = np.arange(len(feature_names))
    axes[0].bar(x, mean_imp, yerr=std_imp, capsize=5,
                color='steelblue', alpha=0.8, edgecolor='navy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[0].set_xlabel('Feature')
    axes[0].set_ylabel('Importance (Mean ± Std)')
    axes[0].set_title(f'SHAP Value Stability (n={results["n_runs"]} runs)')
    axes[0].grid(True, alpha=0.3, axis='y')

    # CV bar plot
    cv = results['cv_importance']
    colors = ['green' if c < 0.1 else 'orange' if c < 0.3 else 'red' for c in cv]
    axes[1].bar(x, cv, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Coefficient of Variation')
    axes[1].set_title('Feature Stability (CV: lower is better)')
    axes[1].axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
    axes[1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate (<0.3)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'SHAP Stability Analysis - Top-K Consistency: {results["top_k_consistency"]:.1f}%',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_dir / 'stability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'stability_analysis.png'}")


# =============================================================================
# PRUNING EFFICIENCY ANALYSIS
# =============================================================================

def analyze_pruning_efficiency(
    model_fn: Callable,
    X_sample: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    pruning_ratios: List[float] = [0.0, 0.25, 0.5, 0.75],
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Analyze trade-off between pruning speedup and accuracy retention.

    Args:
        model_fn: Model prediction function
        X_sample: Input sample
        y_true: True target values
        feature_names: Feature names
        pruning_ratios: Pruning ratios to test
        output_dir: Output directory

    Returns:
        DataFrame with efficiency metrics
    """
    seq_len, n_features = X_sample.shape
    results = []

    # Get full importance as reference
    full_results = _simple_importance_analysis(model_fn, X_sample, feature_names, seq_len)
    full_importance = full_results['feature_importance']

    print("\n" + "=" * 60)
    print("PRUNING EFFICIENCY ANALYSIS")
    print("=" * 60)

    base_runtime = None

    for ratio in pruning_ratios:
        top_k = max(1, int(seq_len * (1 - ratio)))

        start = time.time()
        pruned_results = _simple_importance_analysis(
            model_fn, X_sample, feature_names, top_k
        )
        runtime = time.time() - start

        if ratio == 0.0:
            base_runtime = runtime

        # Accuracy: correlation with full importance
        pruned_importance = pruned_results['feature_importance']
        correlation = np.corrcoef(full_importance, pruned_importance)[0, 1]
        accuracy_retained = max(0, correlation) * 100

        speedup = base_runtime / runtime if runtime > 0 else 1.0

        results.append({
            'pruning_ratio': ratio,
            'top_k': top_k,
            'runtime': runtime,
            'speedup': speedup,
            'accuracy_retained': accuracy_retained,
        })

        print(f"\nPruning {ratio*100:.0f}%: speedup={speedup:.2f}x, accuracy={accuracy_retained:.1f}%")

    df = pd.DataFrame(results)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / 'pruning_efficiency.csv', index=False)
        _plot_pruning_efficiency(df, output_dir)

    return df


def _plot_pruning_efficiency(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot pruning efficiency trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot: speedup vs accuracy
    scatter = ax.scatter(df['speedup'], df['accuracy_retained'],
                         s=200, c=df['pruning_ratio'], cmap='RdYlGn_r',
                         edgecolors='black', linewidths=1.5, alpha=0.8)

    # Add labels
    for _, row in df.iterrows():
        ax.annotate(f"{row['pruning_ratio']*100:.0f}%",
                    (row['speedup'], row['accuracy_retained']),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, fontweight='bold')

    # Find optimal point (knee of curve)
    efficiency_score = df['speedup'] * (df['accuracy_retained'] / 100)
    best_idx = efficiency_score.idxmax()
    best_point = df.iloc[best_idx]

    ax.scatter(best_point['speedup'], best_point['accuracy_retained'],
               s=300, marker='*', color='gold', edgecolors='black',
               linewidths=2, zorder=5, label=f"Optimal ({best_point['pruning_ratio']*100:.0f}%)")

    ax.set_xlabel('Speedup (x)', fontsize=12)
    ax.set_ylabel('Accuracy Retained (%)', fontsize=12)
    ax.set_title('Pruning Efficiency Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pruning Ratio')

    plt.tight_layout()
    plt.savefig(output_dir / 'pruning_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'pruning_efficiency.png'}")


def print_timeshap_summary(results: Dict):
    """Print TimeSHAP summary."""
    print("\n" + "=" * 60)
    print("TIMESHAP ANALYSIS (with Pruning)")
    print("=" * 60)

    print("\nTop timesteps:", sorted(results['top_timesteps']))

    print("\nFeature Importance:")
    for i, (fname, imp) in enumerate(zip(results['feature_names'], results['feature_importance'])):
        print(f"  {fname}: {imp:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    print("Testing TimeSHAP analysis with runtime and stability...")

    # Fake model
    def fake_model(x):
        return x.sum(axis=(1, 2)) if x.ndim == 3 else x.sum()

    # Fake data
    np.random.seed(42)
    X = np.random.randn(168, 5)
    feature_names = ['IMF', 'Delta_X', 'Abs_Delta_X', 'Rolling_Std', 'Rolling_Zscore']

    output_dir = Path(__file__).parent / "test_output"

    # Test basic analysis
    print("\n1. Basic TimeSHAP Analysis:")
    results = run_timeshap_analysis(
        fake_model, X, feature_names,
        top_k_timesteps=10, output_dir=output_dir
    )
    print_timeshap_summary(results)

    # Test runtime measurement
    print("\n2. Runtime Measurement:")
    X_batch = np.random.randn(5, 168, 5)
    runtime_df = measure_shap_runtime(
        fake_model, X_batch, feature_names,
        pruning_ratios=[0.0, 0.5, 0.75],
        n_samples=3, output_dir=output_dir
    )
    print(runtime_df)

    # Test stability analysis
    print("\n3. Stability Analysis:")
    stability_results = analyze_shap_stability(
        fake_model, X, feature_names,
        n_runs=5, top_k_timesteps=10, output_dir=output_dir
    )

    # Test pruning efficiency
    print("\n4. Pruning Efficiency:")
    y_true = np.random.randn(10)
    efficiency_df = analyze_pruning_efficiency(
        fake_model, X, y_true, feature_names,
        pruning_ratios=[0.0, 0.25, 0.5, 0.75],
        output_dir=output_dir
    )
    print(efficiency_df)

    print(f"\nAll test outputs saved to: {output_dir}")
