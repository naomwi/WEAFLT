"""
Training Module for CEEMDAN-based Forecasting
Simple training loop - NO advanced features (no weighted loss, no complex schedulers)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from config import (
    TRAIN_CONFIG, MODEL_CONFIG, DATA_CONFIG, CEEMDAN_CONFIG,
    MODEL_DIR, IMF_DIR, AVAILABLE_MODELS
)
from utils.data_loader import IMFDataManager, load_raw_data, get_target_data
from utils.ceemdan_decomposition import decompose_and_save, load_imfs
from utils.metrics import calculate_all_metrics, print_metrics
from models.lstm import LSTMModel
from models.patchtst import PatchTST
from models.transformer import TransformerModel


# Model registry
MODEL_CLASSES = {
    'lstm': LSTMModel,
    'patchtst': PatchTST,
    'transformer': TransformerModel,
}


def get_device():
    """Get available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_model(model_name: str, seq_len: int, pred_len: int) -> nn.Module:
    """
    Create model instance.

    Args:
        model_name: Name of model ('lstm', 'patchtst', 'transformer')
        seq_len: Input sequence length
        pred_len: Prediction horizon

    Returns:
        Model instance
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CLASSES.keys())}")

    model_class = MODEL_CLASSES[model_name]
    model = model_class(
        input_dim=1,
        output_dim=1,
        seq_len=seq_len,
        pred_len=pred_len
    )

    return model


def train_single_component(
    train_loader,
    val_loader,
    model: nn.Module,
    device: torch.device,
    component_name: str,
    model_name: str,
    pred_len: int
) -> Tuple[nn.Module, Dict]:
    """
    Train model on a single component (IMF or residue).
    Simple training with MSE loss and early stopping.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Model to train
        device: Training device
        component_name: Name of component (e.g., 'imf_1', 'residue')
        model_name: Name of model type
        pred_len: Prediction horizon

    Returns:
        Tuple of (trained model, training history)
    """
    model = model.to(device)

    # Simple MSE loss - NO weighted loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])

    epochs = TRAIN_CONFIG['epochs']
    patience = TRAIN_CONFIG['early_stopping_patience']

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\nTraining {model_name} for {component_name} (horizon={pred_len})...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_state)

    return model, history


def evaluate_component(
    model: nn.Module,
    test_loader,
    device: torch.device,
    data_manager: IMFDataManager,
    component_name: str
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        data_manager: Data manager for inverse scaling
        component_name: Name of component

    Returns:
        Tuple of (y_true, y_pred, metrics)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    # Concatenate batches
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Flatten for metrics
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    # Inverse scale
    preds_original = data_manager.inverse_scale(preds_flat, component_name)
    targets_original = data_manager.inverse_scale(targets_flat, component_name)

    # Calculate metrics
    metrics = calculate_all_metrics(targets_original, preds_original)

    return targets_original, preds_original, metrics


def train_all_components(
    model_name: str,
    pred_len: int,
    device: torch.device,
    imf_data: Dict,
    save_models: bool = True
) -> Dict:
    """
    Train separate models for each IMF and residue.

    Args:
        model_name: Name of model type
        pred_len: Prediction horizon
        device: Training device
        imf_data: Dictionary with 'imfs' and 'residue'
        save_models: Whether to save trained models

    Returns:
        Dictionary with trained models and results
    """
    n_imfs = len(imf_data['imfs'])
    seq_len = DATA_CONFIG['seq_len']
    batch_size = DATA_CONFIG['batch_size']

    results = {
        'models': {},
        'data_managers': {},
        'metrics': {},
    }

    # Train model for each IMF
    for i in range(n_imfs):
        component_name = f'imf_{i+1}'
        component_data = imf_data['imfs'][i]

        # Create data manager and loaders
        data_manager = IMFDataManager()
        train_loader, val_loader, test_loader = data_manager.create_dataloaders(
            component_data, component_name, seq_len, pred_len, batch_size
        )

        # Create and train model
        model = create_model(model_name, seq_len, pred_len)
        model, history = train_single_component(
            train_loader, val_loader, model, device,
            component_name, model_name, pred_len
        )

        # Evaluate
        y_true, y_pred, metrics = evaluate_component(
            model, test_loader, device, data_manager, component_name
        )

        print(f"  {component_name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

        # Save results
        results['models'][component_name] = model
        results['data_managers'][component_name] = data_manager
        results['metrics'][component_name] = metrics

        # Save model
        if save_models:
            model_path = MODEL_DIR / f"{model_name}_{component_name}_h{pred_len}.pth"
            torch.save(model.state_dict(), model_path)

    # Train model for residue
    component_name = 'residue'
    component_data = imf_data['residue']

    data_manager = IMFDataManager()
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        component_data, component_name, seq_len, pred_len, batch_size
    )

    model = create_model(model_name, seq_len, pred_len)
    model, history = train_single_component(
        train_loader, val_loader, model, device,
        component_name, model_name, pred_len
    )

    y_true, y_pred, metrics = evaluate_component(
        model, test_loader, device, data_manager, component_name
    )

    print(f"  {component_name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    results['models'][component_name] = model
    results['data_managers'][component_name] = data_manager
    results['metrics'][component_name] = metrics

    if save_models:
        model_path = MODEL_DIR / f"{model_name}_{component_name}_h{pred_len}.pth"
        torch.save(model.state_dict(), model_path)

    return results


if __name__ == "__main__":
    # Quick test
    print("Testing training module...")

    device = get_device()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Generate synthetic data for testing
    n_samples = 2000
    imf_data = {
        'imfs': [np.sin(np.linspace(0, 10*np.pi, n_samples) * (i+1)) + 0.1*np.random.randn(n_samples)
                 for i in range(3)],
        'residue': np.linspace(0, 1, n_samples) + 0.01*np.random.randn(n_samples)
    }
    imf_data['imfs'] = np.array(imf_data['imfs'])

    print("\nTesting with synthetic data (3 IMFs + residue)...")
    results = train_all_components('lstm', pred_len=24, device=device, imf_data=imf_data, save_models=False)

    print("\nTraining complete!")
    for comp, metrics in results['metrics'].items():
        print(f"  {comp}: MAE={metrics['MAE']:.4f}")
