import math

import pytest

from metrics import evaluate_forecast, mean_absolute_error, root_mean_squared_error


def test_mae_known_value():
    assert mean_absolute_error([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_rmse_known_value():
    assert root_mean_squared_error([1, 2, 3], [2, 2, 5]) == pytest.approx(math.sqrt(5 / 3))


def test_perfect_prediction_is_zero():
    assert evaluate_forecast([1.5, 2.5], [1.5, 2.5]) == {"mae": 0.0, "rmse": 0.0}


def test_rmse_at_least_mae():
    result = evaluate_forecast([0, 0, 0, 10], [1, 1, 1, 1])
    assert result["rmse"] >= result["mae"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        mean_absolute_error([1, 2], [1])


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        root_mean_squared_error([], [])


def test_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        mean_absolute_error([1.0, float("nan")], [1.0, 2.0])


def test_non_numeric_raises():
    with pytest.raises(ValueError, match="numeric"):
        mean_absolute_error(["a", "b"], [1, 2])
