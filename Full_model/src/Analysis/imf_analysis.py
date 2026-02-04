"""
IMF Contribution Analysis Module for CEEMDAN-Enhanced Models.

This module provides tools for analyzing the contribution of each IMF
(Intrinsic Mode Function) to the overall prediction quality.

Key Features:
1. Calculate per-IMF contribution percentages
2. Rank IMFs by importance (via SHAP or variance)
3. Generate contribution reports and visualizations
4. Support for multi-target analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re


class IMFContributionAnalyzer:
    """
    Analyzer for IMF (Intrinsic Mode Function) contributions in CEEMDAN models.

    Supports multiple analysis methods:
    1. Variance-based: IMF contribution based on signal variance
    2. SHAP-based: IMF contribution based on SHAP values (requires shap package)
    3. Ablation-based: IMF contribution based on model performance drop
    """

    def __init__(
        self,
        feature_names: List[str],
        target_names: List[str],
        n_imfs: int = 12,
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            feature_names: List of all feature column names
            target_names: List of target column names
            n_imfs: Number of IMF components per variable
            output_dir: Directory to save reports and visualizations
        """
        self.feature_names = feature_names
        self.target_names = target_names
        self.n_imfs = n_imfs
        self.output_dir = output_dir

        # Parse IMF structure from feature names
        self.imf_mapping = self._parse_imf_structure()
        self.contributions: Dict[str, Dict] = {}

    def _parse_imf_structure(self) -> Dict[str, List[str]]:
        """
        Parse feature names to identify IMF columns for each variable.

        Returns:
            Dictionary mapping variable names to their IMF feature names
        """
        imf_mapping = {}

        # Pattern to match IMF columns: varname_IMF_1, varname_IMF_2, etc.
        imf_pattern = re.compile(r'(.+)_IMF_(\d+)$')
        residue_pattern = re.compile(r'(.+)_residue$')

        for feat in self.feature_names:
            imf_match = imf_pattern.match(feat)
            residue_match = residue_pattern.match(feat)

            if imf_match:
                var_name = imf_match.group(1)
                if var_name not in imf_mapping:
                    imf_mapping[var_name] = {'imfs': [], 'residue': None}
                imf_mapping[var_name]['imfs'].append(feat)
            elif residue_match:
                var_name = residue_match.group(1)
                if var_name not in imf_mapping:
                    imf_mapping[var_name] = {'imfs': [], 'residue': None}
                imf_mapping[var_name]['residue'] = feat

        return imf_mapping

    def analyze_variance_contribution(
        self,
        data: Union[np.ndarray, torch.Tensor, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Analyze IMF contributions based on signal variance.

        Higher variance IMFs typically capture more significant signal components.

        Args:
            data: Input data with IMF features

        Returns:
            DataFrame with variance-based contribution percentages
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(data, pd.DataFrame):
            data = data.values

        results = []

        for var_name, imf_info in self.imf_mapping.items():
            imf_cols = imf_info['imfs']
            residue_col = imf_info['residue']

            # Get indices of IMF columns
            imf_indices = [self.feature_names.index(c) for c in imf_cols if c in self.feature_names]

            if not imf_indices:
                continue

            # Calculate variance for each IMF
            variances = []
            for idx, col_name in zip(imf_indices, imf_cols):
                if data.ndim == 3:
                    # [batch, seq, features]
                    var = np.var(data[:, :, idx])
                else:
                    # [samples, features]
                    var = np.var(data[:, idx])
                variances.append(var)

            # Add residue variance if available
            if residue_col and residue_col in self.feature_names:
                res_idx = self.feature_names.index(residue_col)
                if data.ndim == 3:
                    res_var = np.var(data[:, :, res_idx])
                else:
                    res_var = np.var(data[:, res_idx])
                variances.append(res_var)
                imf_cols = imf_cols + [residue_col]

            # Calculate contribution percentages
            total_var = sum(variances)
            if total_var > 0:
                contributions = [v / total_var * 100 for v in variances]
            else:
                contributions = [0] * len(variances)

            for col, var, contrib in zip(imf_cols, variances, contributions):
                # Determine component type
                if 'residue' in col:
                    comp_type = 'Residue'
                    comp_num = 0
                else:
                    comp_type = 'IMF'
                    match = re.search(r'IMF_(\d+)', col)
                    comp_num = int(match.group(1)) if match else 0

                results.append({
                    'Variable': var_name,
                    'Component': col,
                    'Type': comp_type,
                    'Number': comp_num,
                    'Variance': var,
                    'Contribution_Pct': contrib
                })

        df_results = pd.DataFrame(results)
        self.contributions['variance'] = df_results
        return df_results

    def analyze_shap_contribution(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze IMF contributions based on SHAP values.

        Args:
            shap_values: SHAP values array [samples, seq_len, features] or [samples, features]
            feature_names: Feature names (uses self.feature_names if None)

        Returns:
            DataFrame with SHAP-based contribution analysis
        """
        if feature_names is None:
            feature_names = self.feature_names

        # Aggregate SHAP values (mean absolute value)
        if shap_values.ndim == 3:
            # [samples, seq_len, features] -> [features]
            shap_importance = np.mean(np.abs(shap_values), axis=(0, 1))
        else:
            # [samples, features] -> [features]
            shap_importance = np.mean(np.abs(shap_values), axis=0)

        results = []

        for var_name, imf_info in self.imf_mapping.items():
            imf_cols = imf_info['imfs']
            residue_col = imf_info['residue']

            all_cols = imf_cols.copy()
            if residue_col:
                all_cols.append(residue_col)

            # Get SHAP values for this variable's IMFs
            imf_shap = []
            for col in all_cols:
                if col in feature_names:
                    idx = feature_names.index(col)
                    imf_shap.append(shap_importance[idx])
                else:
                    imf_shap.append(0)

            # Calculate contribution percentages
            total_shap = sum(imf_shap)
            if total_shap > 0:
                contributions = [s / total_shap * 100 for s in imf_shap]
            else:
                contributions = [0] * len(imf_shap)

            for col, shap_val, contrib in zip(all_cols, imf_shap, contributions):
                if 'residue' in col:
                    comp_type = 'Residue'
                    comp_num = 0
                else:
                    comp_type = 'IMF'
                    match = re.search(r'IMF_(\d+)', col)
                    comp_num = int(match.group(1)) if match else 0

                results.append({
                    'Variable': var_name,
                    'Component': col,
                    'Type': comp_type,
                    'Number': comp_num,
                    'SHAP_Importance': shap_val,
                    'Contribution_Pct': contrib
                })

        df_results = pd.DataFrame(results)
        self.contributions['shap'] = df_results
        return df_results

    def get_cumulative_contribution(
        self,
        method: str = 'variance',
        threshold: float = 95.0
    ) -> Tuple[List[str], float]:
        """
        Get features that cumulatively contribute up to a threshold.

        Args:
            method: 'variance' or 'shap'
            threshold: Cumulative contribution threshold (percentage)

        Returns:
            Tuple of (list of important features, actual cumulative contribution)
        """
        if method not in self.contributions:
            raise ValueError(f"No {method} contributions calculated. Run analyze_{method}_contribution first.")

        df = self.contributions[method].copy()

        # Sort by contribution
        df = df.sort_values('Contribution_Pct', ascending=False)

        # Calculate cumulative contribution
        df['Cumulative_Pct'] = df['Contribution_Pct'].cumsum()

        # Get features up to threshold
        important = df[df['Cumulative_Pct'] <= threshold]['Component'].tolist()

        # Always include at least one feature
        if not important:
            important = [df.iloc[0]['Component']]

        actual_contrib = df[df['Component'].isin(important)]['Contribution_Pct'].sum()

        return important, actual_contrib

    def generate_report(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate a comprehensive IMF contribution report.

        Args:
            save_path: Path to save the report CSV

        Returns:
            Combined DataFrame with all contribution analyses
        """
        reports = []

        for method, df in self.contributions.items():
            df_copy = df.copy()
            df_copy['Method'] = method
            reports.append(df_copy)

        if not reports:
            return pd.DataFrame()

        combined = pd.concat(reports, ignore_index=True)

        if save_path:
            combined.to_csv(save_path, index=False)
            print(f"IMF contribution report saved to: {save_path}")

        return combined

    def plot_contribution_heatmap(
        self,
        method: str = 'variance',
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot a heatmap of IMF contributions per variable.

        Args:
            method: 'variance' or 'shap'
            save_path: Path to save the figure
            figsize: Figure size
        """
        if method not in self.contributions:
            print(f"No {method} contributions available.")
            return

        df = self.contributions[method]

        # Pivot to create heatmap data
        pivot_df = df.pivot_table(
            index='Variable',
            columns='Component',
            values='Contribution_Pct',
            aggfunc='first'
        )

        # Sort columns by IMF number
        def sort_key(col):
            if 'residue' in col.lower():
                return 999
            match = re.search(r'IMF_(\d+)', col)
            return int(match.group(1)) if match else 0

        sorted_cols = sorted(pivot_df.columns, key=sort_key)
        pivot_df = pivot_df[sorted_cols]

        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Contribution (%)'}
        )
        plt.title(f'IMF Contribution Analysis ({method.upper()})')
        plt.xlabel('IMF Component')
        plt.ylabel('Variable')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")

        plt.close()


def calculate_imf_contributions(
    data: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str],
    method: str = 'variance'
) -> Dict[str, float]:
    """
    Quick function to calculate IMF contributions.

    Args:
        data: Input data
        feature_names: Feature column names
        method: 'variance' for variance-based contribution

    Returns:
        Dictionary mapping feature names to contribution percentages
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Calculate variance per feature
    if data.ndim == 3:
        variances = np.var(data, axis=(0, 1))
    else:
        variances = np.var(data, axis=0)

    # Filter to IMF columns only
    imf_pattern = re.compile(r'.*_(IMF_\d+|residue)$')
    imf_indices = [i for i, name in enumerate(feature_names) if imf_pattern.match(name)]

    if not imf_indices:
        # Return all features if no IMF pattern found
        total_var = variances.sum()
        return {name: (var / total_var * 100 if total_var > 0 else 0)
                for name, var in zip(feature_names, variances)}

    imf_variances = variances[imf_indices]
    total_var = imf_variances.sum()

    contributions = {}
    for idx in imf_indices:
        name = feature_names[idx]
        contrib = variances[idx] / total_var * 100 if total_var > 0 else 0
        contributions[name] = contrib

    return contributions


def rank_imf_importance(
    contributions: Dict[str, float],
    top_k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Rank IMFs by their importance/contribution.

    Args:
        contributions: Dictionary of feature -> contribution percentage
        top_k: Return only top K features (None for all)

    Returns:
        List of (feature_name, contribution) tuples, sorted by contribution
    """
    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        ranked = ranked[:top_k]

    return ranked


def generate_imf_report(
    analyzer: IMFContributionAnalyzer,
    output_dir: Path,
    include_plots: bool = True
) -> Path:
    """
    Generate a complete IMF analysis report.

    Args:
        analyzer: IMFContributionAnalyzer instance with computed contributions
        output_dir: Directory to save report files
        include_plots: Whether to generate visualization plots

    Returns:
        Path to the main report file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate CSV report
    report_path = output_dir / 'imf_contribution_report.csv'
    df_report = analyzer.generate_report(save_path=report_path)

    # Generate plots
    if include_plots and 'variance' in analyzer.contributions:
        analyzer.plot_contribution_heatmap(
            method='variance',
            save_path=output_dir / 'imf_variance_heatmap.png'
        )

    if include_plots and 'shap' in analyzer.contributions:
        analyzer.plot_contribution_heatmap(
            method='shap',
            save_path=output_dir / 'imf_shap_heatmap.png'
        )

    # Generate summary statistics
    summary_path = output_dir / 'imf_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("IMF Contribution Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        for method, df in analyzer.contributions.items():
            f.write(f"\n{method.upper()} Analysis:\n")
            f.write("-" * 30 + "\n")

            # Top contributors
            top_5 = df.nlargest(5, 'Contribution_Pct')[['Component', 'Contribution_Pct']]
            f.write("Top 5 contributors:\n")
            for _, row in top_5.iterrows():
                f.write(f"  {row['Component']}: {row['Contribution_Pct']:.2f}%\n")

            # 95% threshold
            important, total = analyzer.get_cumulative_contribution(method, 95.0)
            f.write(f"\nFeatures for 95% contribution: {len(important)}\n")
            f.write(f"Actual contribution: {total:.2f}%\n")

    print(f"IMF analysis report generated at: {output_dir}")
    return report_path
