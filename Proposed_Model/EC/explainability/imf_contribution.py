"""
IMF Contribution Analysis
Analyze how each IMF contributes to the final prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_imf_contributions(
    imf_predictions: List[np.ndarray],
    residue_prediction: np.ndarray,
    actual_signal: np.ndarray,
    event_flags: np.ndarray = None,
    output_dir: Path = None
) -> Dict:
    """
    Analyze contribution of each IMF to the final prediction.

    Args:
        imf_predictions: List of predictions from each IMF model
        residue_prediction: Prediction from residue model
        actual_signal: Ground truth signal
        event_flags: Optional event flags for event-specific analysis
        output_dir: Directory to save plots

    Returns:
        Dictionary with contribution analysis
    """
    n_imfs = len(imf_predictions)
    all_preds = imf_predictions + [residue_prediction]
    component_names = [f"IMF_{i+1}" for i in range(n_imfs)] + ["Residue"]

    # Total prediction (sum of all components)
    total_pred = np.sum([p for p in all_preds], axis=0)

    # Calculate contribution of each component
    contributions = []
    for pred in all_preds:
        # Contribution = |pred| / |total|
        contrib = np.mean(np.abs(pred)) / np.mean(np.abs(total_pred)) * 100
        contributions.append(contrib)

    # Normalize to sum to 100%
    total_contrib = sum(contributions)
    contributions = [c / total_contrib * 100 for c in contributions]

    # Calculate contribution at event timesteps
    event_contributions = None
    if event_flags is not None:
        event_idx = np.where(event_flags == 1)[0]
        if len(event_idx) > 0:
            event_contributions = []
            total_at_events = np.mean(np.abs(total_pred[event_idx]))
            for pred in all_preds:
                contrib = np.mean(np.abs(pred[event_idx])) / total_at_events * 100
                event_contributions.append(contrib)
            # Normalize
            total_event = sum(event_contributions)
            event_contributions = [c / total_event * 100 for c in event_contributions]

    # Calculate error contribution (which IMF introduces most error)
    error_contributions = []
    for pred in all_preds:
        # Ablation: remove this component and measure error change
        ablated_pred = total_pred - pred
        error_with = np.mean(np.abs(actual_signal - total_pred))
        error_without = np.mean(np.abs(actual_signal - ablated_pred))
        error_contrib = (error_without - error_with) / error_with * 100
        error_contributions.append(error_contrib)

    results = {
        'component_names': component_names,
        'contributions': contributions,
        'event_contributions': event_contributions,
        'error_contributions': error_contributions,
    }

    # Create summary DataFrame
    df = pd.DataFrame({
        'Component': component_names,
        'Contribution (%)': contributions,
        'Error Impact (%)': error_contributions,
    })
    if event_contributions:
        df['Event Contribution (%)'] = event_contributions

    results['summary_df'] = df

    # Plot if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Contribution chart
        ax1 = axes[0]
        bars = ax1.bar(component_names, contributions, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Contribution (%)')
        ax1.set_title('IMF Contribution to Final Prediction')
        ax1.set_xticklabels(component_names, rotation=45, ha='right')

        # Error impact chart
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in error_contributions]
        ax2.bar(component_names, error_contributions, color=colors, alpha=0.8)
        ax2.set_ylabel('Error Impact (%)')
        ax2.set_title('Error Impact (positive = helpful)')
        ax2.set_xticklabels(component_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / 'imf_contribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save CSV
        df.to_csv(output_dir / 'imf_contribution.csv', index=False)

    return results


def print_contribution_summary(results: Dict):
    """Print contribution summary."""
    print("\n" + "=" * 60)
    print("IMF CONTRIBUTION ANALYSIS")
    print("=" * 60)

    df = results['summary_df']
    print(df.to_string(index=False))

    # Find most important IMFs
    contributions = results['contributions']
    top_idx = np.argsort(contributions)[-3:][::-1]
    names = results['component_names']

    print(f"\nTop 3 contributors: {', '.join([names[i] for i in top_idx])}")
    print("=" * 60)


if __name__ == "__main__":
    print("Testing IMF contribution analysis...")

    # Fake data
    np.random.seed(42)
    n_samples = 100
    n_imfs = 12

    imf_preds = [np.random.randn(n_samples) * (0.5 ** i) for i in range(n_imfs)]
    residue_pred = np.random.randn(n_samples) * 0.1
    actual = sum(imf_preds) + residue_pred + np.random.randn(n_samples) * 0.1
    event_flags = (np.random.rand(n_samples) > 0.95).astype(float)

    results = analyze_imf_contributions(imf_preds, residue_pred, actual, event_flags)
    print_contribution_summary(results)
