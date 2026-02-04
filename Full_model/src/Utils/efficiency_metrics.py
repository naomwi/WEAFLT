"""
Efficiency Metrics Module for Model Performance Analysis.

This module provides tools for measuring computational efficiency:
1. Parameter counting
2. Inference latency
3. Memory profiling
4. FLOPs estimation
5. Pruning efficiency tracking
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import gc


@dataclass
class EfficiencyReport:
    """Container for efficiency metrics."""
    model_name: str
    params_total: int
    params_trainable: int
    inference_time_ms: float
    memory_mb: float
    flops_estimate: int
    throughput: float  # samples/second


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure model inference time.

    Args:
        model: PyTorch model
        input_tensor: Sample input tensor
        n_iterations: Number of timing iterations
        warmup_iterations: Number of warmup iterations
        device: Device to run on

    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_iterations):
            _ = model(input_tensor)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Timing runs
        for _ in range(n_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_tensor)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'per_sample_ms': float(np.mean(times) / input_tensor.shape[0])
    }


def measure_memory_usage(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure GPU memory usage during inference.

    Args:
        model: PyTorch model
        input_tensor: Sample input tensor
        device: Device to measure on ('cuda' only)

    Returns:
        Dictionary with memory statistics (in MB)
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return {
            'allocated_mb': 0,
            'reserved_mb': 0,
            'peak_mb': 0
        }

    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Measure baseline
    baseline_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024

    with torch.no_grad():
        _ = model(input_tensor)

    # Measure after inference
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    current_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024

    return {
        'baseline_mb': baseline_allocated,
        'allocated_mb': current_allocated,
        'reserved_mb': reserved,
        'peak_mb': peak_allocated,
        'inference_mb': peak_allocated - baseline_allocated
    }


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> int:
    """
    Estimate FLOPs (Floating Point Operations) for a model.

    This is a rough estimate based on layer types.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, seq_len, features)
        device: Device to analyze on

    Returns:
        Estimated FLOPs count
    """
    total_flops = 0

    def count_linear_flops(layer, input_size):
        # FLOPs = 2 * input_features * output_features (multiply-add)
        return 2 * layer.in_features * layer.out_features * np.prod(input_size[:-1])

    def count_conv1d_flops(layer, input_size):
        # FLOPs = 2 * kernel_size * in_channels * out_channels * output_length
        batch, seq_len, _ = input_size
        output_len = (seq_len - layer.kernel_size[0]) // layer.stride[0] + 1
        return 2 * layer.kernel_size[0] * layer.in_channels * layer.out_channels * output_len * batch

    def count_lstm_flops(layer, input_size):
        # LSTM: 4 gates, each is a linear operation
        batch, seq_len, input_dim = input_size
        hidden_size = layer.hidden_size
        num_layers = layer.num_layers
        directions = 2 if layer.bidirectional else 1

        # Per timestep: 4 * (input_dim + hidden_size) * hidden_size * 2 (multiply-add)
        flops_per_step = 4 * (input_dim + hidden_size) * hidden_size * 2
        return flops_per_step * seq_len * batch * num_layers * directions

    def count_attention_flops(d_model, seq_len, batch_size):
        # Q, K, V projections: 3 * seq_len * d_model * d_model
        # Attention scores: seq_len * seq_len * d_model
        # Output projection: seq_len * d_model * d_model
        return batch_size * (3 * seq_len * d_model * d_model +
                            seq_len * seq_len * d_model +
                            seq_len * d_model * d_model)

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            # Rough estimate assuming input covers all positions
            total_flops += 2 * layer.in_features * layer.out_features * input_shape[0]
        elif isinstance(layer, nn.LSTM):
            total_flops += count_lstm_flops(layer, input_shape)
        elif isinstance(layer, nn.TransformerEncoderLayer):
            d_model = layer.self_attn.embed_dim
            total_flops += count_attention_flops(d_model, input_shape[1], input_shape[0])
        elif isinstance(layer, nn.MultiheadAttention):
            d_model = layer.embed_dim
            total_flops += count_attention_flops(d_model, input_shape[1], input_shape[0])

    return total_flops


def generate_efficiency_report(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    n_iterations: int = 100
) -> EfficiencyReport:
    """
    Generate a comprehensive efficiency report for a model.

    Args:
        model: PyTorch model
        model_name: Name for identification
        input_tensor: Sample input tensor
        device: Device to run on
        n_iterations: Iterations for timing

    Returns:
        EfficiencyReport dataclass
    """
    # Parameter count
    params_total = count_parameters(model, trainable_only=False)
    params_trainable = count_parameters(model, trainable_only=True)

    # Inference time
    timing = measure_inference_time(model, input_tensor, n_iterations, device=device)
    inference_time = timing['mean_ms']

    # Memory (GPU only)
    if device == 'cuda' and torch.cuda.is_available():
        memory = measure_memory_usage(model, input_tensor, device)
        memory_mb = memory['peak_mb']
    else:
        memory_mb = 0

    # FLOPs estimate
    flops = estimate_flops(model, input_tensor.shape, device)

    # Throughput
    batch_size = input_tensor.shape[0]
    throughput = batch_size / (inference_time / 1000) if inference_time > 0 else 0

    return EfficiencyReport(
        model_name=model_name,
        params_total=params_total,
        params_trainable=params_trainable,
        inference_time_ms=inference_time,
        memory_mb=memory_mb,
        flops_estimate=flops,
        throughput=throughput
    )


def compare_efficiency(
    reports: List[EfficiencyReport],
    baseline_name: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare efficiency across multiple models.

    Args:
        reports: List of EfficiencyReport objects
        baseline_name: Name of baseline model for relative comparison

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}

    # Find baseline
    baseline = None
    if baseline_name:
        for r in reports:
            if r.model_name == baseline_name:
                baseline = r
                break

    if baseline is None and reports:
        baseline = reports[0]

    for report in reports:
        metrics = {
            'params': report.params_trainable,
            'inference_ms': report.inference_time_ms,
            'memory_mb': report.memory_mb,
            'throughput': report.throughput,
            'flops': report.flops_estimate
        }

        if baseline and baseline.model_name != report.model_name:
            # Calculate relative metrics
            metrics['params_relative'] = report.params_trainable / baseline.params_trainable if baseline.params_trainable > 0 else 0
            metrics['speed_relative'] = baseline.inference_time_ms / report.inference_time_ms if report.inference_time_ms > 0 else 0
            metrics['param_reduction_pct'] = (1 - metrics['params_relative']) * 100
            metrics['speedup'] = metrics['speed_relative']

        comparison[report.model_name] = metrics

    return comparison


def format_efficiency_table(comparison: Dict[str, Dict[str, float]]) -> str:
    """
    Format efficiency comparison as a readable table.

    Args:
        comparison: Dictionary from compare_efficiency()

    Returns:
        Formatted table string
    """
    lines = [
        "=" * 80,
        "MODEL EFFICIENCY COMPARISON",
        "=" * 80,
        f"{'Model':<20} {'Params':>12} {'Time (ms)':>12} {'Throughput':>12} {'Speedup':>10}",
        "-" * 80
    ]

    for model_name, metrics in comparison.items():
        params = f"{metrics['params']:,}"
        time_ms = f"{metrics['inference_ms']:.2f}"
        throughput = f"{metrics['throughput']:.1f}"
        speedup = f"{metrics.get('speedup', 1.0):.2f}x"

        lines.append(f"{model_name:<20} {params:>12} {time_ms:>12} {throughput:>12} {speedup:>10}")

    lines.append("=" * 80)

    return "\n".join(lines)
