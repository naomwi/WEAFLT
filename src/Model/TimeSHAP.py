import torch
import torch.nn 
import numpy as np


def temporal_attribution_enhanced(model, dataset, input_window, baseline_window,
                                   segment_size=4, device="cpu"):
    """
    🔧 FIXED: TimeSHAP-compliant baseline (mean across ALL features, not just channel 0)

    Temporal attribution using mean-baseline masking.
    Baseline is computed as the temporal mean across all features.
    """
    model.eval()

    if input_window.ndim == 1:
        input_window = input_window.reshape(-1, 1)
    if baseline_window.ndim == 1:
        baseline_window = baseline_window.reshape(-1, 1)

    T = input_window.shape[0]
    attributions_magnitude = []

    x = torch.FloatTensor(input_window).unsqueeze(0).to(device)
    with torch.no_grad():
        y_full = model(x)
        y_full_denorm = dataset.denormalize(y_full).cpu().numpy().flatten()

    for start in range(0, T, segment_size):
        end = min(start + segment_size, T)
        x_masked = input_window.copy()
        x_masked[start:end] = baseline_window[start:end]

        x_m = torch.FloatTensor(x_masked).unsqueeze(0).to(device)
        with torch.no_grad():
            y_mask = model(x_m)
            y_mask_denorm = dataset.denormalize(y_mask).cpu().numpy().flatten()

        diff = y_full_denorm - y_mask_denorm
        magnitude = np.mean(np.abs(diff))

        attributions_magnitude.append(magnitude)

    return np.array(attributions_magnitude)