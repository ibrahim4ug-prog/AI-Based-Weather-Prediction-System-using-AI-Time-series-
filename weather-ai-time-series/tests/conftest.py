"""Shared deterministic fixtures."""
import numpy as np
import pandas as pd
import pytest

from data_processing import generate_synthetic_weather


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """400 days of deterministic synthetic weather."""
    return generate_synthetic_weather(start="2023-01-01", periods=400, seed=7)


@pytest.fixture
def temperature_series() -> pd.Series:
    """Deterministic, stationary-ish daily series with a regular DatetimeIndex."""
    rng = np.random.default_rng(123)
    n = 200
    noise = np.zeros(n)
    eps = rng.normal(0.0, 1.0, n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + eps[i]
    index = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(25.0 + noise, index=index, name="temperature")
