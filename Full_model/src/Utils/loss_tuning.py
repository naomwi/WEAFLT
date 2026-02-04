"""
Alpha Tuning Module for Event-Weighted Loss.

This module provides tools for automatically tuning the alpha parameter
in the EventWeightedMSE loss function for optimal event detection performance.

Strategies:
1. Grid Search: Test predefined alpha values
2. Bayesian Optimization: Use Optuna for efficient search
3. Adaptive: Adjust alpha during training based on event metrics
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class AlphaTuningResult:
    """Container for alpha tuning results."""
    best_alpha: float
    best_metric: float
    all_results: Dict[float, Dict[str, float]]
    search_time_s: float
    metric_name: str


class AlphaTuner:
    """
    Alpha parameter tuner for EventWeightedMSE loss.

    The alpha parameter controls how much more the loss penalizes
    errors during detected events vs normal periods.

    Formula: weights = 1.0 + (is_event * (alpha - 1.0))
    - alpha=1.0: Standard MSE (no event weighting)
    - alpha=2.0: 2x weight on events
    - alpha=5.0: 5x weight on events
    - alpha=10.0: 10x weight on events
    """

    def __init__(
        self,
        alpha_range: List[float] = None,
        metric: str = 'SF_MAE',
        minimize: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            alpha_range: List of alpha values to try
            metric: Metric to optimize ('SF_MAE', 'F1_Score', etc.)
            minimize: If True, minimize metric; if False, maximize
            verbose: Print progress information
        """
        self.alpha_range = alpha_range or [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        self.metric = metric
        self.minimize = minimize
        self.verbose = verbose
        self.results_history: List[AlphaTuningResult] = []

    def grid_search(
        self,
        train_func: Callable,
        eval_func: Callable,
        model_factory: Callable,
        train_loader,
        val_loader,
        device: str = 'cpu',
        epochs_per_trial: int = 20
    ) -> AlphaTuningResult:
        """
        Perform grid search over alpha values.

        Args:
            train_func: Function(model, criterion, loader, epochs) -> trained_model
            eval_func: Function(model, loader) -> metrics_dict
            model_factory: Function() -> new model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            epochs_per_trial: Epochs per alpha trial

        Returns:
            AlphaTuningResult with best alpha and metrics
        """
        from src.Utils.support_class import EventWeightedMSE

        start_time = time.time()
        all_results = {}

        best_alpha = self.alpha_range[0]
        best_metric_value = float('inf') if self.minimize else float('-inf')

        if self.verbose:
            print(f"\nAlpha Grid Search: Testing {len(self.alpha_range)} values...")
            print(f"Optimizing for: {self.metric} ({'min' if self.minimize else 'max'})")
            print("-" * 50)

        for alpha in self.alpha_range:
            if self.verbose:
                print(f"  Testing alpha={alpha}...", end=" ")

            # Create new model
            model = model_factory().to(device)

            # Create criterion with this alpha
            criterion = EventWeightedMSE(alpha=alpha).to(device)

            # Train
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model = self._quick_train(
                model, criterion, optimizer, train_loader,
                epochs_per_trial, device
            )

            # Evaluate
            metrics = eval_func(model, val_loader, device)
            metric_value = metrics.get(self.metric, float('inf'))

            all_results[alpha] = metrics

            if self.verbose:
                print(f"{self.metric}={metric_value:.4f}")

            # Check if best
            is_better = (metric_value < best_metric_value) if self.minimize else (metric_value > best_metric_value)
            if is_better:
                best_metric_value = metric_value
                best_alpha = alpha

        search_time = time.time() - start_time

        if self.verbose:
            print("-" * 50)
            print(f"Best alpha: {best_alpha} ({self.metric}={best_metric_value:.4f})")
            print(f"Search time: {search_time:.1f}s")

        result = AlphaTuningResult(
            best_alpha=best_alpha,
            best_metric=best_metric_value,
            all_results=all_results,
            search_time_s=search_time,
            metric_name=self.metric
        )

        self.results_history.append(result)
        return result

    def _quick_train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        train_loader,
        epochs: int,
        device: str
    ) -> nn.Module:
        """Quick training without early stopping for tuning."""
        model.train()

        for epoch in range(epochs):
            for x, y, evt, _ in train_loader:
                x, y, evt = x.to(device), y.to(device), evt.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y, evt)
                loss.backward()
                optimizer.step()

        return model

    def bayesian_search(
        self,
        objective_func: Callable,
        n_trials: int = 20,
        alpha_min: float = 1.0,
        alpha_max: float = 20.0
    ) -> AlphaTuningResult:
        """
        Perform Bayesian optimization using Optuna.

        Args:
            objective_func: Function(alpha) -> metric_value
            n_trials: Number of trials
            alpha_min: Minimum alpha value
            alpha_max: Maximum alpha value

        Returns:
            AlphaTuningResult with best alpha
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("Optuna not installed. Falling back to grid search.")
            return self.grid_search_simple(objective_func)

        start_time = time.time()
        all_results = {}

        def optuna_objective(trial):
            alpha = trial.suggest_float('alpha', alpha_min, alpha_max)
            metric_value = objective_func(alpha)
            all_results[alpha] = {self.metric: metric_value}
            return metric_value

        direction = 'minimize' if self.minimize else 'maximize'
        study = optuna.create_study(direction=direction)
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=self.verbose)

        search_time = time.time() - start_time

        result = AlphaTuningResult(
            best_alpha=study.best_params['alpha'],
            best_metric=study.best_value,
            all_results=all_results,
            search_time_s=search_time,
            metric_name=self.metric
        )

        self.results_history.append(result)
        return result

    def grid_search_simple(
        self,
        objective_func: Callable[[float], float]
    ) -> AlphaTuningResult:
        """
        Simple grid search with a callable objective.

        Args:
            objective_func: Function(alpha) -> metric_value

        Returns:
            AlphaTuningResult
        """
        start_time = time.time()
        all_results = {}

        best_alpha = self.alpha_range[0]
        best_metric = float('inf') if self.minimize else float('-inf')

        for alpha in self.alpha_range:
            if self.verbose:
                print(f"  Testing alpha={alpha}...", end=" ")

            metric_value = objective_func(alpha)
            all_results[alpha] = {self.metric: metric_value}

            if self.verbose:
                print(f"{self.metric}={metric_value:.4f}")

            is_better = (metric_value < best_metric) if self.minimize else (metric_value > best_metric)
            if is_better:
                best_metric = metric_value
                best_alpha = alpha

        search_time = time.time() - start_time

        return AlphaTuningResult(
            best_alpha=best_alpha,
            best_metric=best_metric,
            all_results=all_results,
            search_time_s=search_time,
            metric_name=self.metric
        )


class AdaptiveAlphaScheduler:
    """
    Adaptive alpha scheduler that adjusts during training.

    Increases alpha when event detection performance is poor,
    decreases when overfitting to events.
    """

    def __init__(
        self,
        initial_alpha: float = 3.0,
        min_alpha: float = 1.0,
        max_alpha: float = 20.0,
        adjustment_factor: float = 1.2,
        patience: int = 5
    ):
        """
        Args:
            initial_alpha: Starting alpha value
            min_alpha: Minimum allowed alpha
            max_alpha: Maximum allowed alpha
            adjustment_factor: Multiply/divide factor for adjustments
            patience: Epochs before adjustment
        """
        self.alpha = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.adjustment_factor = adjustment_factor
        self.patience = patience

        self.history = []
        self.epochs_without_improvement = 0
        self.best_sf_mae = float('inf')

    def step(self, sf_mae: float, regular_mae: float) -> float:
        """
        Update alpha based on metrics.

        Args:
            sf_mae: Sudden Fluctuation MAE
            regular_mae: Regular MAE

        Returns:
            Updated alpha value
        """
        self.history.append({
            'alpha': self.alpha,
            'sf_mae': sf_mae,
            'regular_mae': regular_mae
        })

        # Calculate ratio
        ratio = sf_mae / (regular_mae + 1e-6)

        # If SF_MAE is much worse than regular MAE, increase alpha
        if ratio > 1.5:
            self.alpha = min(self.max_alpha, self.alpha * self.adjustment_factor)
        # If SF_MAE improved significantly
        elif sf_mae < self.best_sf_mae * 0.95:
            self.best_sf_mae = sf_mae
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            # If no improvement for a while, try adjusting alpha
            if self.epochs_without_improvement >= self.patience:
                # Try decreasing alpha to prevent overfitting
                self.alpha = max(self.min_alpha, self.alpha / self.adjustment_factor)
                self.epochs_without_improvement = 0

        return self.alpha

    def get_history(self) -> List[Dict]:
        """Get history of alpha adjustments."""
        return self.history


def tune_alpha_quick(
    model_factory: Callable,
    train_loader,
    val_loader,
    eval_func: Callable,
    device: str = 'cpu',
    alphas: List[float] = None
) -> Tuple[float, Dict]:
    """
    Quick alpha tuning function for experiments.

    Args:
        model_factory: Function to create new model
        train_loader: Training data
        val_loader: Validation data
        eval_func: Function(model, loader, device) -> metrics
        device: Device to use
        alphas: Alpha values to test

    Returns:
        Tuple of (best_alpha, all_results)
    """
    from src.Utils.support_class import EventWeightedMSE

    alphas = alphas or [1.0, 3.0, 5.0, 10.0]
    results = {}

    for alpha in alphas:
        print(f"  Testing alpha={alpha}...")

        model = model_factory().to(device)
        criterion = EventWeightedMSE(alpha=alpha).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Quick training (10 epochs)
        model.train()
        for _ in range(10):
            for x, y, evt, _ in train_loader:
                x, y, evt = x.to(device), y.to(device), evt.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y, evt)
                loss.backward()
                optimizer.step()

        # Evaluate
        metrics = eval_func(model, val_loader, device)
        results[alpha] = metrics

    # Find best based on SF_MAE
    best_alpha = min(results, key=lambda a: results[a].get('SF_MAE', float('inf')))

    return best_alpha, results
