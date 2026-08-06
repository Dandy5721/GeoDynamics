"""Small numerical helpers shared by the dataset loaders."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _corrcoef(sequence: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    centered = sequence - sequence.mean(axis=-1, keepdims=True)
    covariance = centered @ centered.swapaxes(-2, -1)
    variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    scale = np.sqrt(np.maximum(variance, epsilon))
    correlation = covariance / scale[..., :, None] / scale[..., None, :]
    return np.clip(correlation, -1.0, 1.0)


def sliding_window_corrcoef(
    sequence: np.ndarray, window_size: int, padding: bool = True
) -> np.ndarray:
    """Return one correlation matrix per time point.

    ``sequence`` is shaped ``(features, time)``. Edge padding repeats the first
    and last observed frames, preserving the original sequence length.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"expected (features,time), got {sequence.shape}")
    if not 1 < window_size <= sequence.shape[-1]:
        raise ValueError("window size must be between 2 and the sequence length")
    if padding:
        left = (window_size - 1) // 2
        right = window_size - 1 - left
        sequence = np.pad(sequence, ((0, 0), (left, right)), mode="edge")
    windows = sliding_window_view(sequence, window_size, axis=-1)
    windows = np.moveaxis(windows, 1, 0)
    return _corrcoef(windows)
