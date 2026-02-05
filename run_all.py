#!/usr/bin/env python
"""
Main runner script for Water Quality Forecasting Models
Uses only EC and pH features from USGS data

This script runs all four model directories:

1. Baselines_model: Basic CEEMDAN-LTSF (DLinear, NLinear only)
2. ceemdan_features_model: CEEMDAN + rolling features (DLinear, NLinear)
3. ceemdan_EVloss: CEEMDAN + event-weighted loss (DLinear, NLinear)
4. Full_model: Complete pipeline with ALL models:
   - Linear: DLinear, NLinear, LTSF_Linear, RLinear
   - Deep Learning: LSTM, Transformer
   - Patch-based: PatchTST
   - CEEMDAN variants: CEEMDAN-LSTM, CEEMDAN-Transformer, CEEMDAN-PatchTST
   - Statistical: ARIMA

To run ALL models including PatchTST, LSTM, Transformer:
    python run_all.py --model full

Or run Full_model directly:
    cd Full_model && python main.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Model directories
MODELS = {
    'baselines': ROOT_DIR / 'Baselines_model',
    'features': ROOT_DIR / 'ceemdan_features_model',
    'eventloss': ROOT_DIR / 'ceemdan_EVloss',
    'full': ROOT_DIR / 'Full_model',
}


def run_model(model_name: str, verbose: bool = True) -> int:
    """
    Run a specific model.

    Args:
        model_name: Name of the model to run ('baselines', 'features', 'eventloss', 'full')
        verbose: Whether to print output

    Returns:
        Return code (0 for success)
    """
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(MODELS.keys())}")
        return 1

    model_dir = MODELS[model_name]
    main_script = model_dir / 'main.py'

    if not main_script.exists():
        print(f"Error: main.py not found at {main_script}")
        return 1

    print(f"\n{'='*60}")
    print(f"Running: {model_name.upper()}")
    print(f"Directory: {model_dir}")
    print(f"{'='*60}\n")

    # Run the model
    result = subprocess.run(
        [sys.executable, str(main_script)],
        cwd=str(model_dir),
        capture_output=not verbose,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running {model_name}")
        if not verbose and result.stderr:
            print(result.stderr)
        return result.returncode

    print(f"\n>>> {model_name.upper()} completed successfully!")
    return 0


def run_all_models(verbose: bool = True) -> dict:
    """
    Run all models sequentially.

    Returns:
        Dictionary with model names as keys and return codes as values
    """
    results = {}

    print("\n" + "=" * 70)
    print("WATER QUALITY FORECASTING - Running All Models")
    print("Using only EC and pH features from USGS data")
    print("=" * 70)

    for model_name in MODELS.keys():
        results[model_name] = run_model(model_name, verbose)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name, return_code in results.items():
        status = "SUCCESS" if return_code == 0 else f"FAILED (code: {return_code})"
        print(f"  {model_name}: {status}")

    total_success = sum(1 for rc in results.values() if rc == 0)
    print(f"\nTotal: {total_success}/{len(results)} models completed successfully")

    return results


def run_visualization_only():
    """Run only the visualization scripts for all models."""
    print("\n" + "=" * 70)
    print("Running Visualization Scripts Only")
    print("=" * 70)

    for model_name, model_dir in MODELS.items():
        if model_name == 'full':
            continue  # Full model has integrated visualization

        plot_script = model_dir / 'plot_visual.py'
        if plot_script.exists():
            print(f"\n>>> Visualizing {model_name}...")
            subprocess.run(
                [sys.executable, str(plot_script)],
                cwd=str(model_dir)
            )


def main():
    parser = argparse.ArgumentParser(
        description='Run Water Quality Forecasting Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Run all 4 model directories
  python run_all.py --model baselines  # Run Baselines_model (DLinear, NLinear)
  python run_all.py --model features   # Run ceemdan_features_model
  python run_all.py --model eventloss  # Run ceemdan_EVloss
  python run_all.py --model full       # Run Full_model (ALL models incl. PatchTST, LSTM, Transformer)
  python run_all.py --visualize        # Run visualization only
  python run_all.py --list             # List available models

Model Details:
  baselines  : CEEMDAN + DLinear/NLinear (basic)
  features   : CEEMDAN + rolling features + DLinear/NLinear
  eventloss  : CEEMDAN + event-weighted loss + DLinear/NLinear
  full       : ALL models (DLinear, NLinear, RLinear, PatchTST, LSTM, Transformer, ARIMA)
               + CEEMDAN variants + XAI analysis + IMF pruning
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=list(MODELS.keys()),
        help='Specific model to run (default: run all)'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Run visualization scripts only'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available models'
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, path in MODELS.items():
            print(f"  {name}: {path}")
        return 0

    if args.visualize:
        run_visualization_only()
        return 0

    if args.model:
        return run_model(args.model, verbose=not args.quiet)
    else:
        results = run_all_models(verbose=not args.quiet)
        # Return non-zero if any model failed
        return 0 if all(rc == 0 for rc in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
