"""ARIMA forecasting of a single weather variable (temperature) with statsmodels.

Only temperature is modelled. No other weather parameter has its own model,
so none is forecast. Forecasts are statistical estimates; their quality depends
on the data and on the model, and plain ARIMA does not model annual seasonality.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metrics import evaluate_forecast

MIN_OBSERVATIONS = 30
MIN_TRAIN_OBSERVATIONS = 20
MIN_TEST_OBSERVATIONS = 3
MAX_HORIZON = 365
Order = Tuple[int, int, int]


class ForecastingError(Exception):
    """Base class for all forecasting problems shown to the user."""


class InsufficientDataError(ForecastingError):
    """Not enough (or degenerate) data for reliable forecasting."""


class InvalidForecastInputError(ForecastingError):
    """Invalid horizon, split, order, confidence level or series."""


class ModelFitError(ForecastingError):
    """The ARIMA model could not be fitted or produced invalid output."""


@dataclass
class ForecastResult:
    """Everything produced by :func:`run_forecast`."""

    variable: str
    order: Order
    aic: float
    train: pd.Series
    test: pd.Series
    test_predictions: pd.DataFrame      # columns: forecast, lower, upper (index = test dates)
    future_forecast: pd.DataFrame       # columns: forecast, lower, upper (index = future dates)
    metrics: Dict[str, float]           # mae, rmse of the ARIMA model on the test set
    baseline_metrics: Dict[str, float]  # mae, rmse of "repeat last training value"
    confidence_level: float
    horizon: int
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation and splitting
# --------------------------------------------------------------------------- #
def validate_series(series: pd.Series) -> pd.Series:
    """Validate and normalise a series for modelling (sorted, regular, finite)."""
    if not isinstance(series, pd.Series):
        raise InvalidForecastInputError("The input must be a pandas Series.")
    if series.empty:
        raise InsufficientDataError("The series is empty; there is nothing to forecast.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise InvalidForecastInputError("The series must have a DatetimeIndex.")
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise InvalidForecastInputError(
            "The series contains missing or non-finite values. Clean the data first."
        )
    if values.index.has_duplicates:
        raise InvalidForecastInputError("The series has duplicate dates. Clean the data first.")
    values = values.sort_index()  # guarantees chronological order before any split
    if len(values) < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"At least {MIN_OBSERVATIONS} observations are required for ARIMA forecasting; "
            f"only {len(values)} available."
        )
    if float(values.std()) == 0.0:
        raise InsufficientDataError("The series is constant; a time-series model cannot be fitted.")
    if values.index.freq is None:
        try:
            inferred = pd.infer_freq(values.index)
        except (TypeError, ValueError):
            inferred = None
        if inferred is None:
            raise InvalidForecastInputError(
                "The series is not evenly spaced. Use data_processing.make_regular_series first."
            )
        values = values.asfreq(inferred)
    return values


def chronological_split(
    series: pd.Series,
    test_fraction: float = 0.2,
    test_size: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Split into (train, test) by time: train = earliest, test = most recent.

    No shuffling is performed, so no test observation precedes a training one.
    """
    s = validate_series(series)
    n = len(s)
    if test_size is None:
        if not 0.0 < float(test_fraction) < 1.0:
            raise InvalidForecastInputError("test_fraction must be between 0 and 1 (exclusive).")
        test_size = max(MIN_TEST_OBSERVATIONS, int(round(n * float(test_fraction))))
    test_size = int(test_size)
    train_size = n - test_size
    if test_size < MIN_TEST_OBSERVATIONS or train_size < MIN_TRAIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"A split of {n} observations leaves {train_size} for training and {test_size} for testing; "
            f"at least {MIN_TRAIN_OBSERVATIONS} and {MIN_TEST_OBSERVATIONS} are required."
        )
    return s.iloc[:train_size], s.iloc[train_size:]


def validate_order(order: Any) -> Order:
    """Validate an ARIMA (p, d, q) order."""
    try:
        p, d, q = (int(v) for v in order)
        if len(tuple(order)) != 3:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise InvalidForecastInputError("order must be three integers (p, d, q).") from exc
    if not (0 <= p <= 10 and 0 <= d <= 2 and 0 <= q <= 10):
        raise InvalidForecastInputError("order must satisfy 0<=p<=10, 0<=d<=2, 0<=q<=10.")
    return p, d, q


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def _import_arima() -> Any:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ModelFitError(
            "statsmodels is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return ARIMA


def fit_arima(train: pd.Series, order: Any) -> Any:
    """Fit ARIMA(p, d, q) on ``train`` and return the statsmodels results object."""
    p_d_q = validate_order(order)
    ARIMA = _import_arima()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ARIMA(train, order=p_d_q).fit()
    except Exception as exc:  # statsmodels raises ValueError, LinAlgError, ...
        raise ModelFitError(f"ARIMA{p_d_q} could not be fitted: {exc}") from exc


def choose_differencing(train: pd.Series, alpha: float = 0.05) -> int:
    """Pick d=0 if the Augmented Dickey-Fuller test suggests stationarity, else d=1."""
    try:
        from statsmodels.tsa.stattools import adfuller

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_value = float(adfuller(train.to_numpy(), autolag="AIC")[1])
        return 0 if p_value < alpha else 1
    except Exception:
        return 1


def select_order(train: pd.Series, max_p: int = 2, max_q: int = 2) -> Tuple[Order, float]:
    """Choose (p, d, q) by lowest AIC over a small grid (d from an ADF test)."""
    d = choose_differencing(train)
    best: Optional[Tuple[Order, float]] = None
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            aic = float(fit_arima(train, (p, d, q)).aic)
        except ModelFitError:
            continue
        if np.isfinite(aic) and (best is None or aic < best[1]):
            best = ((p, d, q), aic)
    if best is None:
        raise ModelFitError("No ARIMA configuration could be fitted to this series.")
    return best


def forecast_with_intervals(fitted: Any, steps: int, alpha: float = 0.05) -> pd.DataFrame:
    """Forecast ``steps`` ahead; returns columns forecast, lower, upper."""
    try:
        res = fitted.get_forecast(steps=steps)
        mean = res.predicted_mean
        ci = res.conf_int(alpha=alpha)
        out = pd.DataFrame(
            {
                "forecast": np.asarray(mean, dtype=float),
                "lower": np.asarray(ci.iloc[:, 0], dtype=float),
                "upper": np.asarray(ci.iloc[:, 1], dtype=float),
            },
            index=mean.index,
        )
    except Exception as exc:
        raise ModelFitError(f"Forecast generation failed: {exc}") from exc
    if not np.isfinite(out.to_numpy()).all():
        raise ModelFitError("The model produced non-finite forecasts; try a different order.")
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def run_forecast(
    series: pd.Series,
    horizon: int = 14,
    test_fraction: float = 0.2,
    order: Optional[Any] = None,
    confidence_level: float = 0.95,
    variable: str = "temperature",
) -> ForecastResult:
    """Train on the earliest data, evaluate on the latest, then forecast ahead.

    1. Split chronologically (train first, test last).
    2. Choose the order on the TRAIN data only (unless ``order`` is given).
    3. Fit on train and forecast the whole test period -> MAE/RMSE.
    4. Refit on all data with the same order and forecast ``horizon`` steps.

    Raises:
        ForecastingError subclasses for invalid input, insufficient data or fit failures.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise InvalidForecastInputError("The forecast horizon must be a whole number of steps.")
    if not 1 <= int(horizon) <= MAX_HORIZON:
        raise InvalidForecastInputError(f"The forecast horizon must be between 1 and {MAX_HORIZON}.")
    if not 0.5 < float(confidence_level) < 1.0:
        raise InvalidForecastInputError("confidence_level must be between 0.5 and 1 (exclusive).")
    horizon = int(horizon)
    alpha = 1.0 - float(confidence_level)

    try:
        full = validate_series(series)
        train, test = chronological_split(full, test_fraction=test_fraction)
        if order is None:
            chosen, aic = select_order(train)
            fitted_train = fit_arima(train, chosen)
        else:
            chosen = validate_order(order)
            fitted_train = fit_arima(train, chosen)
            aic = float(fitted_train.aic)

        test_fc = forecast_with_intervals(fitted_train, len(test), alpha)
        test_fc.index = test.index
        metrics = evaluate_forecast(test.to_numpy(), test_fc["forecast"].to_numpy())
        baseline = evaluate_forecast(test.to_numpy(), np.full(len(test), float(train.iloc[-1])))

        fitted_full = fit_arima(full, chosen)
        future = forecast_with_intervals(fitted_full, horizon, alpha)
    except ForecastingError:
        raise
    except Exception as exc:
        raise ModelFitError(f"Unexpected forecasting error: {exc}") from exc

    notes: List[str] = []
    if horizon > len(test):
        notes.append(
            f"The horizon ({horizon}) is longer than the evaluated test period ({len(test)}); "
            "accuracy beyond the tested range is unknown."
        )
    if metrics["rmse"] >= baseline["rmse"]:
        notes.append(
            "The ARIMA model did not beat the naive baseline (repeat the last training value) "
            "on the test period, so its added value is limited for this data."
        )
    notes.append(
        "Plain ARIMA does not model annual seasonality; forecasts several weeks or more ahead "
        "tend to flatten towards a level and prediction intervals widen."
    )
    return ForecastResult(
        variable=variable,
        order=chosen,
        aic=float(aic),
        train=train,
        test=test,
        test_predictions=test_fc,
        future_forecast=future,
        metrics=metrics,
        baseline_metrics=baseline,
        confidence_level=float(confidence_level),
        horizon=horizon,
        warnings=notes,
    )


def future_forecast_frame(result: ForecastResult) -> pd.DataFrame:
    """Future forecast as a flat DataFrame suitable for CSV export."""
    out = result.future_forecast.copy()
    out.index.name = "date"
    out = out.reset_index().rename(
        columns={"forecast": "forecast_temperature_c", "lower": "lower_bound_c", "upper": "upper_bound_c"}
    )
    out.insert(1, "confidence_level", result.confidence_level)
    return out


def actual_vs_predicted_frame(result: ForecastResult) -> pd.DataFrame:
    """Actual vs predicted values on the test period as a flat DataFrame."""
    out = result.test_predictions.copy()
    out.insert(0, "actual", result.test.to_numpy())
    out.index.name = "date"
    out = out.reset_index().rename(
        columns={"actual": "actual_temperature_c", "forecast": "predicted_temperature_c",
                 "lower": "lower_bound_c", "upper": "upper_bound_c"}
    )
    out["error_c"] = out["predicted_temperature_c"] - out["actual_temperature_c"]
    return out
