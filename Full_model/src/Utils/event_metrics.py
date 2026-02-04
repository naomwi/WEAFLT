"""
Event Detection Metrics Module for Water Quality Forecasting.

This module provides comprehensive metrics for evaluating event detection
performance, including sudden fluctuation detection in water quality parameters.

Metrics Included:
1. Classification metrics: Precision, Recall, F1-score
2. Ranking metrics: AUROC, AUPRC
3. Custom metrics: SF_MAE (Sudden Fluctuation MAE), Event Window Accuracy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve
)


def calculate_event_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_flags: np.ndarray,
    threshold: float = 0.5,
    event_detection_threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive event detection metrics.

    Args:
        y_true: True values [samples, pred_len, targets] or flattened
        y_pred: Predicted values [samples, pred_len, targets] or flattened
        event_flags: Binary event indicators (same shape as y_true)
        threshold: Threshold for error-based event detection
        event_detection_threshold: If set, use prediction errors to detect events

    Returns:
        Dictionary with all event detection metrics
    """
    # Flatten arrays if needed
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    event_flags = np.array(event_flags).flatten()

    # Ensure binary event flags
    event_true = (event_flags > 0.5).astype(int)

    metrics = {}

    # Basic event statistics
    n_events = np.sum(event_true)
    n_total = len(event_true)
    metrics['n_events'] = int(n_events)
    metrics['event_ratio'] = n_events / n_total if n_total > 0 else 0

    if n_events == 0 or n_events == n_total:
        # Cannot compute metrics if all same class
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1_score'] = 0.0
        metrics['auroc'] = 0.5
        metrics['auprc'] = n_events / n_total
        metrics['mcc'] = 0.0
        return metrics

    # Calculate prediction error
    pred_error = np.abs(y_true - y_pred)

    # If using error-based event detection
    if event_detection_threshold is not None:
        # Predict event if error exceeds threshold
        event_pred = (pred_error > event_detection_threshold).astype(int)
    else:
        # Use median error as threshold
        error_threshold = np.percentile(pred_error, 80)
        event_pred = (pred_error > error_threshold).astype(int)

    # Classification metrics
    metrics['precision'] = precision_score(event_true, event_pred, zero_division=0)
    metrics['recall'] = recall_score(event_true, event_pred, zero_division=0)
    metrics['f1_score'] = f1_score(event_true, event_pred, zero_division=0)

    # Matthews Correlation Coefficient (good for imbalanced data)
    metrics['mcc'] = matthews_corrcoef(event_true, event_pred)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(event_true, event_pred, labels=[0, 1]).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)

    # Rates
    metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # AUROC (using prediction error as score)
    try:
        metrics['auroc'] = roc_auc_score(event_true, pred_error)
    except ValueError:
        metrics['auroc'] = 0.5

    # AUPRC (Area Under Precision-Recall Curve)
    try:
        metrics['auprc'] = average_precision_score(event_true, pred_error)
    except ValueError:
        metrics['auprc'] = n_events / n_total

    return metrics


def calculate_sf_mae_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_flags: np.ndarray,
    severity_thresholds: Tuple[float, float] = (0.5, 1.0)
) -> Dict[str, float]:
    """
    Calculate detailed Sudden Fluctuation MAE metrics.

    Breaks down SF_MAE by event severity:
    - Mild: events with small magnitude changes
    - Moderate: events with medium magnitude changes
    - Severe: events with large magnitude changes

    Args:
        y_true: True values
        y_pred: Predicted values
        event_flags: Binary event indicators
        severity_thresholds: (mild_threshold, severe_threshold) for classifying severity

    Returns:
        Dictionary with detailed SF_MAE metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    event_flags = np.array(event_flags).flatten()

    metrics = {}

    # Get event indices
    event_idx = np.where(event_flags > 0.5)[0]
    non_event_idx = np.where(event_flags <= 0.5)[0]

    # Overall SF_MAE
    if len(event_idx) > 0:
        metrics['sf_mae'] = np.mean(np.abs(y_true[event_idx] - y_pred[event_idx]))
    else:
        metrics['sf_mae'] = 0.0

    # Non-event MAE for comparison
    if len(non_event_idx) > 0:
        metrics['non_event_mae'] = np.mean(np.abs(y_true[non_event_idx] - y_pred[non_event_idx]))
    else:
        metrics['non_event_mae'] = 0.0

    # Event MAE ratio (how much worse we do on events)
    if metrics['non_event_mae'] > 0:
        metrics['event_mae_ratio'] = metrics['sf_mae'] / metrics['non_event_mae']
    else:
        metrics['event_mae_ratio'] = 1.0

    # Calculate error magnitude at events
    if len(event_idx) > 0:
        event_errors = np.abs(y_true[event_idx] - y_pred[event_idx])

        # Classify by severity based on error percentiles
        mild_thresh, severe_thresh = severity_thresholds
        error_percentiles = np.percentile(event_errors, [33, 66])

        mild_mask = event_errors <= error_percentiles[0]
        moderate_mask = (event_errors > error_percentiles[0]) & (event_errors <= error_percentiles[1])
        severe_mask = event_errors > error_percentiles[1]

        metrics['sf_mae_mild'] = np.mean(event_errors[mild_mask]) if np.any(mild_mask) else 0.0
        metrics['sf_mae_moderate'] = np.mean(event_errors[moderate_mask]) if np.any(moderate_mask) else 0.0
        metrics['sf_mae_severe'] = np.mean(event_errors[severe_mask]) if np.any(severe_mask) else 0.0

        metrics['n_mild_events'] = int(np.sum(mild_mask))
        metrics['n_moderate_events'] = int(np.sum(moderate_mask))
        metrics['n_severe_events'] = int(np.sum(severe_mask))

    return metrics


def calculate_temporal_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_flags: np.ndarray,
    window_size: int = 6
) -> Dict[str, float]:
    """
    Calculate event metrics with temporal context.

    Considers not just point-wise event detection but also the ability
    to predict events within a time window (early warning capability).

    Args:
        y_true: True values [samples, pred_len, targets] or [samples, targets]
        y_pred: Predicted values
        event_flags: Event indicators
        window_size: Time window for early detection credit

    Returns:
        Dictionary with temporal event metrics
    """
    # Reshape if needed
    if y_true.ndim == 3:
        batch, pred_len, targets = y_true.shape
    elif y_true.ndim == 2:
        batch, features = y_true.shape
        pred_len = 1
        targets = features
    else:
        # Flatten case
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        event_flags = event_flags.reshape(-1, 1)
        batch = len(y_true)
        pred_len = 1
        targets = 1

    metrics = {}

    # Per-horizon metrics
    if pred_len > 1:
        for t in range(min(pred_len, 10)):  # Limit to first 10 steps
            if y_true.ndim == 3:
                t_true = y_true[:, t, :].flatten()
                t_pred = y_pred[:, t, :].flatten()
                t_event = event_flags[:, t, :].flatten() if event_flags.ndim == 3 else event_flags.flatten()
            else:
                continue

            event_idx = np.where(t_event > 0.5)[0]
            if len(event_idx) > 0:
                metrics[f'sf_mae_t{t+1}'] = np.mean(np.abs(t_true[event_idx] - t_pred[event_idx]))
            else:
                metrics[f'sf_mae_t{t+1}'] = 0.0

    return metrics


def calculate_all_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_flags: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all event-related metrics in one call.

    Args:
        y_true: True values
        y_pred: Predicted values
        event_flags: Event indicators

    Returns:
        Combined dictionary with all metrics
    """
    metrics = {}

    # Event detection metrics
    detection_metrics = calculate_event_detection_metrics(y_true, y_pred, event_flags)
    metrics.update({f'event_{k}': v for k, v in detection_metrics.items()})

    # Detailed SF_MAE
    sf_metrics = calculate_sf_mae_detailed(y_true, y_pred, event_flags)
    metrics.update(sf_metrics)

    # Temporal metrics
    temporal_metrics = calculate_temporal_event_metrics(y_true, y_pred, event_flags)
    metrics.update(temporal_metrics)

    return metrics


def generate_event_metrics_report(
    metrics: Dict[str, float],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a formatted report of event metrics.

    Args:
        metrics: Dictionary of computed metrics
        output_path: Optional path to save report

    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 60,
        "EVENT DETECTION METRICS REPORT",
        "=" * 60,
        "",
        "Classification Metrics:",
        "-" * 30,
        f"  Precision:     {metrics.get('event_precision', 0):.4f}",
        f"  Recall:        {metrics.get('event_recall', 0):.4f}",
        f"  F1-Score:      {metrics.get('event_f1_score', 0):.4f}",
        f"  MCC:           {metrics.get('event_mcc', 0):.4f}",
        "",
        "Ranking Metrics:",
        "-" * 30,
        f"  AUROC:         {metrics.get('event_auroc', 0):.4f}",
        f"  AUPRC:         {metrics.get('event_auprc', 0):.4f}",
        "",
        "Sudden Fluctuation MAE:",
        "-" * 30,
        f"  Overall:       {metrics.get('sf_mae', 0):.4f}",
        f"  Mild Events:   {metrics.get('sf_mae_mild', 0):.4f}",
        f"  Moderate:      {metrics.get('sf_mae_moderate', 0):.4f}",
        f"  Severe:        {metrics.get('sf_mae_severe', 0):.4f}",
        "",
        "Event Statistics:",
        "-" * 30,
        f"  Total Events:  {metrics.get('event_n_events', 0)}",
        f"  Event Ratio:   {metrics.get('event_event_ratio', 0):.4f}",
        f"  TP/FP/TN/FN:   {metrics.get('event_true_positives', 0)}/{metrics.get('event_false_positives', 0)}/"
        f"{metrics.get('event_true_negatives', 0)}/{metrics.get('event_false_negatives', 0)}",
        "",
        "=" * 60,
    ]

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report
