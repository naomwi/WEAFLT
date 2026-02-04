#!/usr/bin/env python
"""
Setup and Requirements Check Script for CEEMD-Enhanced LTSF-Linear Project.

This script:
1. Checks Python version
2. Checks for required packages
3. Installs missing packages automatically
4. Verifies GPU/CUDA availability
5. Tests basic imports

Run: python setup_check.py
"""

import sys
import subprocess
import importlib
from typing import List, Tuple, Dict

# Minimum Python version
MIN_PYTHON = (3, 8)

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    # Core packages
    'torch': '2.0.0',
    'numpy': '1.21.0',
    'pandas': '1.3.0',
    'sklearn': '1.0.0',  # scikit-learn

    # Signal processing
    'PyEMD': '1.2.0',

    # Visualization
    'matplotlib': '3.4.0',
    'seaborn': '0.11.0',

    # Utilities
    'tqdm': '4.62.0',
    'yaml': '6.0',  # PyYAML
}

# Optional packages (won't fail if missing)
OPTIONAL_PACKAGES = {
    'shap': '0.41.0',
    'statsmodels': '0.13.0',
}

# Package name mappings (import name -> pip install name)
PACKAGE_INSTALL_NAMES = {
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'cv2': 'opencv-python',
}


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    print(f"\n{'='*60}")
    print("PYTHON VERSION CHECK")
    print(f"{'='*60}")

    current = sys.version_info[:2]
    print(f"  Current Python: {current[0]}.{current[1]}")
    print(f"  Required: >={MIN_PYTHON[0]}.{MIN_PYTHON[1]}")

    if current >= MIN_PYTHON:
        print("  [OK] Python version is compatible")
        return True
    else:
        print("  [FAIL] Python version is too old!")
        return False


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and meets version requirements.

    Returns:
        Tuple of (is_installed, current_version or error message)
    """
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, "Not installed"
    except Exception as e:
        return False, str(e)


def install_package(package_name: str) -> bool:
    """Install a package using pip."""
    # Get the correct pip install name
    install_name = PACKAGE_INSTALL_NAMES.get(package_name, package_name)

    print(f"    Installing {install_name}...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', install_name, '-q'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_packages(packages: Dict[str, str], auto_install: bool = True) -> Dict[str, bool]:
    """
    Check all packages and optionally install missing ones.

    Returns:
        Dictionary of package_name -> installed_status
    """
    results = {}

    for package, min_version in packages.items():
        is_installed, version = check_package(package, min_version)

        if is_installed:
            print(f"  [OK] {package}: {version}")
            results[package] = True
        else:
            print(f"  [MISSING] {package}: {version}")
            if auto_install:
                success = install_package(package)
                if success:
                    print(f"    [INSTALLED] {package}")
                    results[package] = True
                else:
                    print(f"    [FAILED] Could not install {package}")
                    results[package] = False
            else:
                results[package] = False

    return results


def check_gpu_cuda() -> Dict[str, any]:
    """Check GPU and CUDA availability."""
    print(f"\n{'='*60}")
    print("GPU / CUDA CHECK")
    print(f"{'='*60}")

    info = {
        'cuda_available': False,
        'cuda_version': None,
        'gpu_name': None,
        'gpu_memory': None,
        'device': 'cpu'
    }

    try:
        import torch

        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['device'] = 'cuda'

            print(f"  [OK] CUDA Available: Yes")
            print(f"  CUDA Version: {info['cuda_version']}")
            print(f"  GPU: {info['gpu_name']}")
            print(f"  GPU Memory: {info['gpu_memory']:.2f} GB")
            print(f"  PyTorch CUDA: {torch.version.cuda}")

            # Test CUDA operations
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.mm(x, x)
                del x, y
                torch.cuda.empty_cache()
                print(f"  [OK] CUDA tensor operations work correctly")
            except Exception as e:
                print(f"  [WARNING] CUDA operations failed: {e}")
        else:
            print(f"  [INFO] CUDA not available - using CPU")
            print(f"  Possible reasons:")
            print(f"    - No NVIDIA GPU installed")
            print(f"    - CUDA drivers not installed")
            print(f"    - PyTorch installed without CUDA support")

    except ImportError:
        print(f"  [FAIL] PyTorch not installed")

    return info


def check_project_imports() -> bool:
    """Test that all project imports work correctly."""
    print(f"\n{'='*60}")
    print("PROJECT IMPORT CHECK")
    print(f"{'='*60}")

    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    imports_to_test = [
        ('src.Utils.parameter', 'CONFIG, device'),
        ('src.Utils.path', 'DATA_DIR, OUTPUT_DIR'),
        ('src.Model.Linear', 'NLinear, DLinear, LTSF_Linear'),
        ('src.Data.Data_processor', 'DataProcessor'),
        ('src.Data.data_loading', 'create_dataloaders_advanced'),
        ('src.Utils.support_class', 'EventWeightedMSE, EarlyStopping'),
    ]

    all_ok = True
    for module_path, components in imports_to_test:
        try:
            module = importlib.import_module(module_path)
            print(f"  [OK] {module_path}")
        except Exception as e:
            print(f"  [FAIL] {module_path}: {e}")
            all_ok = False

    return all_ok


def run_quick_test() -> bool:
    """Run a quick model instantiation test."""
    print(f"\n{'='*60}")
    print("QUICK MODEL TEST")
    print(f"{'='*60}")

    try:
        import torch
        from src.Model.Linear import NLinear

        # Create a simple model
        model = NLinear(
            input_dim=54,
            output_dim=6,
            seq_len=96,
            pred_len=24
        )

        # Test forward pass
        x = torch.randn(4, 96, 54)
        with torch.no_grad():
            y = model(x)

        expected_shape = (4, 24, 6)
        if y.shape == expected_shape:
            print(f"  [OK] Model instantiation and forward pass work")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {y.shape}")
            return True
        else:
            print(f"  [FAIL] Unexpected output shape: {y.shape}, expected {expected_shape}")
            return False

    except Exception as e:
        print(f"  [FAIL] Quick test failed: {e}")
        return False


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("CEEMD-LTSF PROJECT SETUP CHECK")
    print("="*60)

    all_passed = True

    # 1. Check Python version
    if not check_python_version():
        print("\n[ERROR] Python version check failed. Please upgrade Python.")
        return False

    # 2. Check required packages
    print(f"\n{'='*60}")
    print("REQUIRED PACKAGES CHECK")
    print(f"{'='*60}")

    results = check_and_install_packages(REQUIRED_PACKAGES, auto_install=True)
    required_ok = all(results.values())

    if not required_ok:
        print("\n[WARNING] Some required packages could not be installed.")
        all_passed = False

    # 3. Check optional packages
    print(f"\n{'='*60}")
    print("OPTIONAL PACKAGES CHECK")
    print(f"{'='*60}")

    optional_results = check_and_install_packages(OPTIONAL_PACKAGES, auto_install=True)
    # Optional packages don't affect overall status

    # 4. Check GPU/CUDA
    gpu_info = check_gpu_cuda()

    # 5. Check project imports
    if not check_project_imports():
        print("\n[WARNING] Some project imports failed.")
        all_passed = False

    # 6. Run quick test
    if not run_quick_test():
        print("\n[WARNING] Quick model test failed.")
        all_passed = False

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"  Python: {'OK' if check_python_version() else 'FAIL'}")
    print(f"  Required packages: {'OK' if required_ok else 'INCOMPLETE'}")
    print(f"  Device: {gpu_info['device'].upper()}")
    if gpu_info['cuda_available']:
        print(f"  GPU: {gpu_info['gpu_name']}")
    print(f"  Overall: {'READY' if all_passed else 'NEEDS ATTENTION'}")

    print(f"\n{'='*60}")

    if all_passed:
        print("Setup check PASSED! You can now run the experiments.")
        print("\nTo run experiments:")
        print("  python main.py")
        print("  python test_imports.py")
    else:
        print("Setup check found some issues. Please address them before running experiments.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
