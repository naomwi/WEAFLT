#!/usr/bin/env python
"""
Test script to verify all model imports work correctly.
Run from Full_model directory: python test_imports.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all model and utility imports."""
    print("Testing imports...")
    errors = []

    # Test 1: Core utilities
    try:
        from src.Utils.parameter import CONFIG, device
        print(f"  [OK] Utils.parameter - CONFIG has {len(CONFIG)} keys, device={device}")
    except Exception as e:
        errors.append(f"Utils.parameter: {e}")
        print(f"  [FAIL] Utils.parameter: {e}")

    # Test: New modules
    try:
        from src.Utils.event_metrics import calculate_event_detection_metrics, calculate_sf_mae_detailed
        print("  [OK] Utils.event_metrics - event detection metrics")
    except Exception as e:
        errors.append(f"Utils.event_metrics: {e}")
        print(f"  [FAIL] Utils.event_metrics: {e}")

    try:
        from src.Utils.efficiency_metrics import count_parameters, measure_inference_time
        print("  [OK] Utils.efficiency_metrics - efficiency metrics")
    except Exception as e:
        errors.append(f"Utils.efficiency_metrics: {e}")
        print(f"  [FAIL] Utils.efficiency_metrics: {e}")

    try:
        from src.Utils.loss_tuning import AlphaTuner, AdaptiveAlphaScheduler
        print("  [OK] Utils.loss_tuning - alpha tuning")
    except Exception as e:
        errors.append(f"Utils.loss_tuning: {e}")
        print(f"  [FAIL] Utils.loss_tuning: {e}")

    try:
        from src.Analysis.imf_analysis import IMFContributionAnalyzer, calculate_imf_contributions
        print("  [OK] Analysis.imf_analysis - IMF contribution analysis")
    except Exception as e:
        errors.append(f"Analysis.imf_analysis: {e}")
        print(f"  [FAIL] Analysis.imf_analysis: {e}")

    try:
        from src.Analysis.pruning import IMFPruner, prune_low_contribution_features
        print("  [OK] Analysis.pruning - IMF pruning")
    except Exception as e:
        errors.append(f"Analysis.pruning: {e}")
        print(f"  [FAIL] Analysis.pruning: {e}")

    # Test 2: Path utilities
    try:
        from src.Utils.path import DATA_DIR, OUTPUT_DIR
        print(f"  [OK] Utils.path - DATA_DIR={DATA_DIR}, OUTPUT_DIR={OUTPUT_DIR}")
    except Exception as e:
        errors.append(f"Utils.path: {e}")
        print(f"  [FAIL] Utils.path: {e}")

    # Test 3: Linear models
    try:
        from src.Model.Linear import NLinear, DLinear, LTSF_Linear
        print("  [OK] Model.Linear - NLinear, DLinear, LTSF_Linear")
    except Exception as e:
        errors.append(f"Model.Linear: {e}")
        print(f"  [FAIL] Model.Linear: {e}")

    # Test 4: PatchTST
    try:
        from src.Model.patchtst import PatchTST, CEEMDANPatchTST
        print("  [OK] Model.patchtst - PatchTST, CEEMDANPatchTST")
    except Exception as e:
        errors.append(f"Model.patchtst: {e}")
        print(f"  [FAIL] Model.patchtst: {e}")

    # Test 5: Baseline models
    try:
        from src.Model.baselines import LSTMModel, TransformerModel, ARIMAWrapper
        print("  [OK] Model.baselines - LSTMModel, TransformerModel, ARIMAWrapper")
    except Exception as e:
        errors.append(f"Model.baselines: {e}")
        print(f"  [FAIL] Model.baselines: {e}")

    # Test 6: Support classes
    try:
        from src.Utils.support_class import EventWeightedMSE, EarlyStopping
        print("  [OK] Utils.support_class - EventWeightedMSE, EarlyStopping")
    except Exception as e:
        errors.append(f"Utils.support_class: {e}")
        print(f"  [FAIL] Utils.support_class: {e}")

    # Test 7: Data loading
    try:
        from src.Data.data_loading import create_dataloaders_advanced, CEEMDAN_WaterDataset
        print("  [OK] Data.data_loading - create_dataloaders_advanced, CEEMDAN_WaterDataset")
    except Exception as e:
        errors.append(f"Data.data_loading: {e}")
        print(f"  [FAIL] Data.data_loading: {e}")

    # Test 8: Data processor
    try:
        from src.Data.Data_processor import DataProcessor
        print("  [OK] Data.Data_processor - DataProcessor")
    except Exception as e:
        errors.append(f"Data.Data_processor: {e}")
        print(f"  [FAIL] Data.Data_processor: {e}")

    # Test 9: Experiments
    try:
        from src.Experiments.deploy_experiments import (
            get_model, run_single_trial,
            exp_model_comparison, exp_full_baseline_comparison,
            exp_patchtst_analysis
        )
        print("  [OK] Experiments.deploy_experiments - all experiment functions")
    except Exception as e:
        errors.append(f"Experiments.deploy_experiments: {e}")
        print(f"  [FAIL] Experiments.deploy_experiments: {e}")

    # Test 10: XAI
    try:
        from src.Model.TimeSHAP import XAI_Handler, ShapModelWrapper
        print("  [OK] Model.TimeSHAP - XAI_Handler, ShapModelWrapper")
    except Exception as e:
        errors.append(f"Model.TimeSHAP: {e}")
        print(f"  [FAIL] Model.TimeSHAP: {e}")

    # Summary
    print("\n" + "="*50)
    if errors:
        print(f"FAILED: {len(errors)} imports failed")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("SUCCESS: All imports passed!")
        return True


def test_model_instantiation():
    """Test that models can be instantiated."""
    import torch
    from src.Experiments.deploy_experiments import get_model, MODEL_REGISTRY

    print("\nTesting model instantiation...")

    # Test parameters
    input_dim = 54
    output_dim = 6
    seq_len = 96
    pred_len = 24
    device = 'cpu'

    errors = []
    for model_name in MODEL_REGISTRY.keys():
        try:
            model = get_model(model_name, input_dim, output_dim, seq_len, pred_len, device)
            params = sum(p.numel() for p in model.parameters())
            print(f"  [OK] {model_name}: {params:,} parameters")
        except Exception as e:
            errors.append(f"{model_name}: {e}")
            print(f"  [FAIL] {model_name}: {e}")

    print("\n" + "="*50)
    if errors:
        print(f"FAILED: {len(errors)} models failed to instantiate")
        return False
    else:
        print("SUCCESS: All models instantiated!")
        return True


def test_forward_pass():
    """Test forward pass for all models."""
    import torch
    from src.Experiments.deploy_experiments import get_model, MODEL_REGISTRY

    print("\nTesting forward pass...")

    # Test parameters
    batch_size = 4
    input_dim = 54
    output_dim = 6
    seq_len = 96
    pred_len = 24
    device = 'cpu'

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)

    errors = []
    for model_name in MODEL_REGISTRY.keys():
        try:
            model = get_model(model_name, input_dim, output_dim, seq_len, pred_len, device)
            model.eval()
            with torch.no_grad():
                output = model(x)
            expected_shape = (batch_size, pred_len, output_dim)
            if output.shape == expected_shape:
                print(f"  [OK] {model_name}: output shape {output.shape}")
            else:
                errors.append(f"{model_name}: expected {expected_shape}, got {output.shape}")
                print(f"  [FAIL] {model_name}: expected {expected_shape}, got {output.shape}")
        except Exception as e:
            errors.append(f"{model_name}: {e}")
            print(f"  [FAIL] {model_name}: {e}")

    print("\n" + "="*50)
    if errors:
        print(f"FAILED: {len(errors)} models failed forward pass")
        return False
    else:
        print("SUCCESS: All models passed forward test!")
        return True


if __name__ == "__main__":
    print("="*50)
    print("IMPORT AND MODEL VERIFICATION TEST")
    print("="*50 + "\n")

    all_passed = True

    # Run tests
    all_passed &= test_imports()
    all_passed &= test_model_instantiation()
    all_passed &= test_forward_pass()

    print("\n" + "="*50)
    if all_passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
