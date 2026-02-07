"""
Run All Experiments
Chạy tất cả experiments cho cả EC và pH
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).parent

EXPERIMENTS = {
    # Proposed Model
    'Proposed_Model/EC': {
        'models': ['dlinear', 'nlinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon}',
    },
    'Proposed_Model/pH': {
        'models': ['dlinear', 'nlinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon}',
    },

    # CEEMD Baselines
    'CEEMD_Baselines/EC': {
        'models': ['dlinear', 'nlinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon}',
    },
    'CEEMD_Baselines/pH': {
        'models': ['dlinear', 'nlinear'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon}',
    },

    # Deep Baselines
    'Deep_Baselines/EC': {
        'models': ['lstm', 'patchtst', 'transformer'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon}',
    },
    'Deep_Baselines/pH': {
        'models': ['lstm', 'patchtst', 'transformer'],
        'script': 'main.py',
        'arg_format': '--model {model} --horizon {horizon}',
    },
}

HORIZONS = [6, 12, 24, 48, 96, 168]


def run_experiment(folder: str, model: str, horizon: int, verbose: bool = True):
    """Run a single experiment."""
    exp_dir = ROOT_DIR / folder
    script = EXPERIMENTS[folder]['script']
    arg_format = EXPERIMENTS[folder]['arg_format']

    cmd = f"python {script} {arg_format.format(model=model, horizon=horizon)}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {folder} | {model} | h={horizon}")
        print(f"Command: {cmd}")
        print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=exp_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout per experiment
        )

        if result.returncode != 0:
            print(f"ERROR: {result.stderr[:500]}")
            return False

        if verbose:
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"  {line}")

        return True

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {folder}/{model}/h{horizon}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_all(
    targets: list = None,
    models: list = None,
    horizons: list = None,
    folders: list = None,
    verbose: bool = True
):
    """
    Run all experiments.

    Args:
        targets: ['EC', 'pH'] or None for all
        models: ['dlinear', 'lstm', etc.] or None for all
        horizons: [24, 48] or None for all
        folders: ['Proposed_Model', 'CEEMD_Baselines', 'Deep_Baselines'] or None for all
    """
    if targets is None:
        targets = ['EC', 'pH']
    if horizons is None:
        horizons = HORIZONS
    if folders is None:
        folders = ['Proposed_Model', 'CEEMD_Baselines', 'Deep_Baselines']

    total_experiments = 0
    successful = 0
    failed = []

    start_time = time.time()

    print("\n" + "#" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("#" * 70)
    print(f"Targets: {targets}")
    print(f"Horizons: {horizons}")
    print(f"Folders: {folders}")

    for folder, config in EXPERIMENTS.items():
        # Filter by folder
        folder_base = folder.split('/')[0]
        if folder_base not in folders:
            continue

        # Filter by target
        target = folder.split('/')[1]
        if target not in targets:
            continue

        # Get models for this folder
        exp_models = config['models']
        if models is not None:
            exp_models = [m for m in exp_models if m in models]

        for model in exp_models:
            for horizon in horizons:
                total_experiments += 1
                success = run_experiment(folder, model, horizon, verbose)

                if success:
                    successful += 1
                else:
                    failed.append(f"{folder}/{model}/h{horizon}")

    elapsed = time.time() - start_time

    print("\n" + "#" * 70)
    print("SUMMARY")
    print("#" * 70)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Time: {elapsed/60:.1f} minutes")

    if failed:
        print("\nFailed experiments:")
        for f in failed:
            print(f"  - {f}")

    print("#" * 70)


def main():
    parser = argparse.ArgumentParser(description='Run All Experiments')

    parser.add_argument('--target', '-t', type=str, nargs='+',
                        choices=['EC', 'pH'], default=None,
                        help='Target(s) to run (default: all)')

    parser.add_argument('--model', '-m', type=str, nargs='+', default=None,
                        help='Model(s) to run (default: all)')

    parser.add_argument('--horizon', '-H', type=int, nargs='+', default=None,
                        help='Horizon(s) to run (default: all)')

    parser.add_argument('--folder', '-f', type=str, nargs='+',
                        choices=['Proposed_Model', 'CEEMD_Baselines', 'Deep_Baselines'],
                        default=None, help='Folder(s) to run (default: all)')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (less output)')

    # Quick presets
    parser.add_argument('--proposed-only', action='store_true',
                        help='Run only Proposed Model')
    parser.add_argument('--baselines-only', action='store_true',
                        help='Run only baselines (CEEMD + Deep)')
    parser.add_argument('--ec-only', action='store_true',
                        help='Run only EC target')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: only horizon 24')

    args = parser.parse_args()

    # Apply presets
    targets = args.target
    models = args.model
    horizons = args.horizon
    folders = args.folder

    if args.proposed_only:
        folders = ['Proposed_Model']
    if args.baselines_only:
        folders = ['CEEMD_Baselines', 'Deep_Baselines']
    if args.ec_only:
        targets = ['EC']
    if args.quick:
        horizons = [24]

    run_all(
        targets=targets,
        models=models,
        horizons=horizons,
        folders=folders,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
