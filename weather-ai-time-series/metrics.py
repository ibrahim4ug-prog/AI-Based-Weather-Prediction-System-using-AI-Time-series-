"""Model evaluation metrics for forecasting."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _validated_arrays(actual: ArrayLike, predicted: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and validate that they can be compared.

    Raises:
        ValueError: if either input is empty, the lengths differ, or any value
            is NaN/infinite or not numeric.
    """
    try:
        a = np.asarray(actual, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("actual and predicted values must be numeric.") from exc
    if a.size == 0 or p.size == 0:
        raise ValueError("actual and predicted values must not be empty.")
    if a.shape != p.shape:
        raise ValueError(
            f"actual and predicted must have the same length (got {a.size} and {p.size})."
        )
    if not (np.isfinite(a).all() and np.isfinite(p).all()):
        raise ValueError("actual and predicted values must not contain NaN or infinite values.")
    return a, p


def mean_absolute_error(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Error: average of |actual - predicted| (same unit as the data)."""
    a, p = _validated_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def root_mean_squared_error(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root Mean Squared Error: sqrt(mean((actual - predicted)^2)); penalises large errors."""
    a, p = _validated_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def evaluate_forecast(actual: ArrayLike, predicted: ArrayLike) -> Dict[str, float]:
    """Return ``{"mae": ..., "rmse": ...}`` for the given actual/predicted values."""
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
    }
