"""
Training Module for Proposed Model: CA-CEEMDAN-LTSF
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        x, y, event_flag = batch
        x = x.to(device)
        y = y.to(device)
        event_flag = event_flag.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y, event_flag)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x, y, event_flag = batch
            x = x.to(device)
            y = y.to(device)
            event_flag = event_flag.to(device)

            pred = model(x)
            loss = criterion(pred, y, event_flag)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    verbose: bool = True,
    save_path: Optional[Path] = None
) -> Dict:
    """Train model with early stopping."""
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            history['best_epoch'] = epoch
            history['best_val_loss'] = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'history': history}, save_path)

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Training done in {elapsed:.1f}s, best_val={best_val_loss:.6f}")

    return history


def train_all_imf_models(
    imfs: np.ndarray,
    residue: np.ndarray,
    features: np.ndarray,
    event_flags: np.ndarray,
    model_type: str,
    seq_len: int,
    pred_len: int,
    device: torch.device,
    config: dict,
    save_dir: Path,
    verbose: bool = True
) -> List[nn.Module]:
    """
    Train models for all IMFs + residue.

    Returns:
        List of trained models
    """
    from models import DLinear, NLinear
    from utils.data_loader import create_dataloaders
    from utils.losses import EventWeightedLoss

    n_imfs = len(imfs)
    all_models = []
    all_scalers = []

    # Create criterion
    criterion = EventWeightedLoss(event_weight=config.get('event_weight', 4.0))
    if verbose:
        print(f"Using EventWeightedLoss (event_weight={config.get('event_weight', 4.0)})")

    # Components = IMFs + residue
    components = list(imfs) + [residue]
    names = [f"IMF_{i+1}" for i in range(n_imfs)] + ["Residue"]

    for i, (component, name) in enumerate(zip(components, names)):
        if verbose:
            print(f"\nTraining {name}...")

        # Create dataloaders
        train_loader, val_loader, test_loader, imf_scaler, feat_scaler = create_dataloaders(
            component, features, event_flags,
            seq_len=seq_len, pred_len=pred_len,
            batch_size=config.get('batch_size', 64)
        )

        # Create model
        if model_type == 'dlinear':
            model = DLinear(seq_len=seq_len, pred_len=pred_len, input_dim=5, output_dim=1)
        else:
            model = NLinear(seq_len=seq_len, pred_len=pred_len, input_dim=5, output_dim=1)

        # Train
        save_path = save_dir / f"h{pred_len}" / f"{name.lower()}.pt"
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=config.get('epochs', 50),
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5),
            early_stopping_patience=config.get('early_stopping_patience', 10),
            verbose=False,
            save_path=save_path
        )

        if verbose:
            print(f"  Best val_loss: {history['best_val_loss']:.6f}")

        all_models.append(model)
        all_scalers.append((imf_scaler, feat_scaler))

    return all_models, all_scalers


def train_all_imf_models_parallel(
    imfs: np.ndarray,
    residue: np.ndarray,
    features: np.ndarray,
    event_flags: np.ndarray,
    model_type: str,
    seq_len: int,
    pred_len: int,
    device: torch.device,
    config: dict,
    save_dir: Path,
    max_workers: int = 4,
    verbose: bool = True
) -> Tuple[List[nn.Module], List]:
    """
    Train models for all IMFs + residue in PARALLEL.

    Uses ThreadPoolExecutor to train multiple IMF models simultaneously,
    improving GPU utilization for lightweight models like DLinear/NLinear.

    Args:
        max_workers: Number of parallel training jobs (default: 4)

    Returns:
        Tuple of (list of trained models, list of scalers)
    """
    from models import DLinear, NLinear
    from utils.data_loader import create_dataloaders
    from utils.losses import EventWeightedLoss

    n_imfs = len(imfs)
    components = list(imfs) + [residue]
    names = [f"IMF_{i+1}" for i in range(n_imfs)] + ["Residue"]

    # Thread-safe storage for results
    results = [None] * len(components)
    print_lock = threading.Lock()

    def train_single_imf(idx: int, component: np.ndarray, name: str):
        """Train a single IMF model."""
        # Create dataloaders
        train_loader, val_loader, test_loader, imf_scaler, feat_scaler = create_dataloaders(
            component, features, event_flags,
            seq_len=seq_len, pred_len=pred_len,
            batch_size=config.get('batch_size', 64)
        )

        # Create model
        if model_type == 'dlinear':
            model = DLinear(seq_len=seq_len, pred_len=pred_len, input_dim=5, output_dim=1)
        else:
            model = NLinear(seq_len=seq_len, pred_len=pred_len, input_dim=5, output_dim=1)

        # Loss function (create per-thread to avoid issues)
        criterion = EventWeightedLoss(event_weight=config.get('event_weight', 3.0))

        # Train
        save_path = save_dir / f"h{pred_len}" / f"{name.lower()}.pt"
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=config.get('epochs', 50),
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5),
            early_stopping_patience=config.get('early_stopping_patience', 10),
            verbose=False,
            save_path=save_path
        )

        if verbose:
            with print_lock:
                print(f"  {name}: val_loss={history['best_val_loss']:.6f}")

        return idx, model, (imf_scaler, feat_scaler), history

    if verbose:
        print(f"\nTraining {len(components)} IMF models in parallel (max_workers={max_workers})...")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_single_imf, i, comp, name): i
            for i, (comp, name) in enumerate(zip(components, names))
        }

        for future in as_completed(futures):
            try:
                idx, model, scalers, history = future.result()
                results[idx] = (model, scalers)
            except Exception as e:
                idx = futures[future]
                print(f"  Error training {names[idx]}: {e}")
                raise

    elapsed = time.time() - start_time
    if verbose:
        print(f"  All models trained in {elapsed:.1f}s")

    all_models = [r[0] for r in results]
    all_scalers = [r[1] for r in results]

    return all_models, all_scalers


def train_aggregator(
    aggregator: nn.Module,
    imf_models: List[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    scalers: List,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Train the IMF aggregator to learn optimal attention weights.

    The aggregator is trained end-to-end while keeping IMF models frozen.

    Args:
        aggregator: IMFAggregator model
        imf_models: List of trained per-IMF models (will be frozen)
        train_loader: Training data loader (for original signal)
        val_loader: Validation data loader
        scalers: List of (imf_scaler, feat_scaler) for each IMF
        device: torch device
        epochs: Number of training epochs
        learning_rate: Learning rate for aggregator
        early_stopping_patience: Early stopping patience
        verbose: Print progress

    Returns:
        Dict with training history
    """
    # Freeze IMF models
    for model in imf_models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    aggregator = aggregator.to(device)
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    if verbose:
        print(f"\nTraining aggregator...")

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        aggregator.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)

            # Get predictions from all IMF models
            imf_preds = []
            with torch.no_grad():
                for model in imf_models:
                    pred = model(x)  # (batch, pred_len, 1)
                    imf_preds.append(pred)

            # Aggregate
            optimizer.zero_grad()
            final_pred = aggregator(imf_preds)
            loss = criterion(final_pred, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        history['train_loss'].append(train_loss)

        # Validation
        aggregator.eval()
        val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y, _ = batch
                x = x.to(device)
                y = y.to(device)

                imf_preds = []
                for model in imf_models:
                    pred = model(x)
                    imf_preds.append(pred)

                final_pred = aggregator(imf_preds)
                loss = criterion(final_pred, y.unsqueeze(-1))
                val_loss += loss.item()
                n_batches += 1

        val_loss /= max(n_batches, 1)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            history['best_epoch'] = epoch
            history['best_val_loss'] = val_loss
            best_state = {k: v.cpu().clone() for k, v in aggregator.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        aggregator.load_state_dict(best_state)

    elapsed = time.time() - start_time
    if verbose:
        print(f"    Aggregator trained in {elapsed:.1f}s, best_val={best_val_loss:.6f}")

    return history


if __name__ == "__main__":
    print("Testing training module...")
