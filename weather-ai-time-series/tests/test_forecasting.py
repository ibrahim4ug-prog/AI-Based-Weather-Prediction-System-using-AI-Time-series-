import numpy as np
import pandas as pd
import pytest

from forecasting import (
    InsufficientDataError,
    InvalidForecastInputError,
    chronological_split,
    fit_arima,
    forecast_with_intervals,
    future_forecast_frame,
    actual_vs_predicted_frame,
    run_forecast,
    validate_series,
)


@pytest.fixture
def statsmodels_available():
    return pytest.importorskip("statsmodels")


# ---- splitting (no statsmodels needed) ------------------------------------ #
def test_split_is_chronological_without_overlap(temperature_series):
    train, test = chronological_split(temperature_series, test_fraction=0.2)
    assert len(train) + len(test) == len(temperature_series)
    assert train.index.max() < test.index.min()
    assert test.index.max() == temperature_series.index.max()
    assert len(test) == 40


def test_split_sorts_shuffled_input(temperature_series):
    shuffled = temperature_series.sample(frac=1.0, random_state=0)
    train, test = chronological_split(shuffled, test_fraction=0.25)
    assert train.index.is_monotonic_increasing and train.index.max() < test.index.min()


def test_split_insufficient_data():
    short = pd.Series(np.arange(10.0), index=pd.date_range("2024-01-01", periods=10))
    with pytest.raises(InsufficientDataError, match="At least"):
        chronological_split(short)


def test_split_invalid_fraction(temperature_series):
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(InvalidForecastInputError):
            chronological_split(temperature_series, test_fraction=bad)


def test_validate_rejects_nan_constant_and_wrong_type(temperature_series):
    with_nan = temperature_series.copy()
    with_nan.iloc[5] = np.nan
    with pytest.raises(InvalidForecastInputError, match="missing"):
        validate_series(with_nan)
    constant = pd.Series(5.0, index=temperature_series.index)
    with pytest.raises(InsufficientDataError, match="constant"):
        validate_series(constant)
    with pytest.raises(InvalidForecastInputError):
        validate_series([1, 2, 3])
    with pytest.raises(InsufficientDataError):
        validate_series(pd.Series(dtype=float, index=pd.DatetimeIndex([])))


# ---- ARIMA (requires statsmodels) ----------------------------------------- #
def test_fit_arima_and_forecast(statsmodels_available, temperature_series):
    fitted = fit_arima(temperature_series, (1, 0, 0))
    fc = forecast_with_intervals(fitted, steps=7, alpha=0.05)
    assert list(fc.columns) == ["forecast", "lower", "upper"]
    assert len(fc) == 7
    assert fc.index[0] == temperature_series.index[-1] + pd.Timedelta(days=1)
    assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()


def test_run_forecast_end_to_end(statsmodels_available, temperature_series):
    result = run_forecast(temperature_series, horizon=10, test_fraction=0.2,
                          order=(1, 0, 1), confidence_level=0.9)
    assert result.order == (1, 0, 1)
    assert len(result.test) == 40 and len(result.train) == 160
    assert result.train.index.max() < result.test.index.min()
    assert len(result.test_predictions) == len(result.test)
    assert (result.test_predictions.index == result.test.index).all()
    assert len(result.future_forecast) == 10
    assert result.future_forecast.index[0] > temperature_series.index[-1]
    assert result.metrics["mae"] >= 0 and result.metrics["rmse"] >= result.metrics["mae"]
    assert np.isfinite(list(result.metrics.values())).all()
    assert set(result.baseline_metrics) == {"mae", "rmse"}
    assert future_forecast_frame(result).shape == (10, 5)
    assert "error_c" in actual_vs_predicted_frame(result).columns


def test_run_forecast_automatic_order(statsmodels_available, temperature_series):
    result = run_forecast(temperature_series, horizon=5)
    p, d, q = result.order
    assert 0 <= p <= 2 and d in (0, 1) and 0 <= q <= 2
    assert np.isfinite(result.aic)


def test_run_forecast_invalid_inputs(temperature_series):
    for bad in (0, -3, 1000, 2.5, "7", True):
        with pytest.raises(InvalidForecastInputError):
            run_forecast(temperature_series, horizon=bad)
    for bad in (0.2, 1.0, 1.5):
        with pytest.raises(InvalidForecastInputError):
            run_forecast(temperature_series, confidence_level=bad)
    for bad in ((-1, 0, 0), (1, 5, 1), (1, 1), "abc"):
        with pytest.raises(InvalidForecastInputError):
            run_forecast(temperature_series, order=bad)


def test_run_forecast_insufficient_data():
    short = pd.Series(np.arange(12.0), index=pd.date_range("2024-01-01", periods=12))
    with pytest.raises(InsufficientDataError):
        run_forecast(short, horizon=3)
