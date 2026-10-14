from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_processing import (
    DataValidationError,
    clean_weather_data,
    compute_overview,
    filter_by_date,
    generate_synthetic_weather,
    load_csv,
    make_regular_series,
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "weather.csv"


def write_csv(tmp_path, text: str):
    path = tmp_path / "data.csv"
    path.write_text(text)
    return path


# ---- CSV loading ---------------------------------------------------------- #
def test_load_csv_valid(tmp_path):
    path = write_csv(tmp_path, "date,temperature\n2024-01-01,20\n2024-01-02,21\n")
    assert len(load_csv(path)) == 2


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_csv(tmp_path / "nope.csv")


def test_load_csv_empty_file(tmp_path):
    with pytest.raises(DataValidationError, match="empty"):
        load_csv(write_csv(tmp_path, ""))


def test_load_csv_header_only(tmp_path):
    with pytest.raises(DataValidationError, match="no data rows"):
        load_csv(write_csv(tmp_path, "date,temperature\n"))


# ---- validation ----------------------------------------------------------- #
def test_missing_required_column_message():
    df = pd.DataFrame({"date": ["2024-01-01"], "humidity": [50]})
    with pytest.raises(DataValidationError, match="temperature"):
        clean_weather_data(df)


def test_column_aliases_and_units_in_headers():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"],
                       "Temp (°C)": [20.0, 21.0],
                       "Wind Speed (km/h)": [5.0, 6.0],
                       "Notes": ["a", "b"]})
    clean, report = clean_weather_data(df)
    assert list(clean.columns) == ["date", "temperature", "wind_speed"]
    assert any("Notes" in w for w in report.warnings)


def test_invalid_dates_removed():
    df = pd.DataFrame({"date": ["2024-01-01", "not-a-date", None, "2024-01-03"],
                       "temperature": [10.0, 11.0, 12.0, 13.0]})
    clean, report = clean_weather_data(df)
    assert len(clean) == 2
    assert report.invalid_dates_removed == 2


def test_all_dates_invalid_raises():
    df = pd.DataFrame({"date": ["x", "y"], "temperature": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="valid dates"):
        clean_weather_data(df)


def test_incorrect_numeric_values_become_missing_then_filled():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3).astype(str),
                       "temperature": [10.0, "abc", 30.0],
                       "humidity": [50.0, 150.0, 70.0]})   # 150 % is impossible
    clean, report = clean_weather_data(df)
    assert report.invalid_numeric_values == 1
    assert report.out_of_range_values == 1
    assert clean["temperature"].tolist() == [10.0, 20.0, 30.0]
    assert clean["humidity"].tolist() == [50.0, 60.0, 70.0]


def test_missing_values_interpolated_and_rainfall_zero():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4).astype(str),
                       "temperature": [10.0, np.nan, np.nan, 40.0],
                       "rainfall": [1.0, np.nan, 2.0, 3.0]})
    clean, report = clean_weather_data(df)
    assert clean["temperature"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert clean["rainfall"].tolist() == [1.0, 0.0, 2.0, 3.0]
    assert report.missing_values_filled == {"temperature": 2, "rainfall": 1}
    assert not clean.isna().any().any()


def test_all_missing_optional_column_dropped():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3).astype(str),
                       "temperature": [1.0, 2.0, 3.0], "humidity": [np.nan] * 3})
    clean, _ = clean_weather_data(df)
    assert "humidity" not in clean.columns


def test_all_temperature_invalid_raises():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3).astype(str),
                       "temperature": ["a", "b", "c"]})
    with pytest.raises(DataValidationError, match="temperature"):
        clean_weather_data(df)


def test_duplicate_dates_keep_last():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-01"],
                       "temperature": [10.0, 20.0, 99.0 - 60.0]})
    clean, report = clean_weather_data(df)
    assert len(clean) == 2
    assert report.duplicate_dates_removed == 1
    assert clean.loc[clean["date"] == "2024-01-01", "temperature"].iloc[0] == 39.0


def test_sorted_chronologically_and_reported():
    df = pd.DataFrame({"date": ["2024-01-03", "2024-01-01", "2024-01-02"],
                       "temperature": [3.0, 1.0, 2.0]})
    clean, report = clean_weather_data(df)
    assert clean["date"].is_monotonic_increasing
    assert clean["temperature"].tolist() == [1.0, 2.0, 3.0]
    assert report.input_was_unsorted


def test_dayfirst_parsing():
    df = pd.DataFrame({"date": ["13/01/2024", "14/01/2024"], "temperature": [1.0, 2.0]})
    clean, _ = clean_weather_data(df, dayfirst=True)
    assert clean["date"].iloc[0] == pd.Timestamp("2024-01-13")


# ---- helpers -------------------------------------------------------------- #
def test_make_regular_series_fills_calendar_gaps():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
                       "temperature": [10.0, 20.0, 50.0]})
    series, info = make_regular_series(df)
    assert len(series) == 5
    assert series.index.freq is not None
    assert info["gaps_filled"] == 2
    assert series.iloc[2] == pytest.approx(30.0)


def test_make_regular_series_missing_column():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "humidity": [1.0]})
    with pytest.raises(DataValidationError):
        make_regular_series(df, "temperature")


def test_filter_by_date_inclusive():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10), "temperature": range(10)})
    out = filter_by_date(df, "2024-01-03", "2024-01-05")
    assert out["temperature"].tolist() == [2, 3, 4]


def test_compute_overview_handles_missing_columns():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "temperature": [1.0, 5.0, 3.0]})
    ov = compute_overview(df)
    assert ov["records"] == 3 and ov["max_temperature"] == 5.0
    assert ov["avg_humidity"] is None and ov["total_rainfall"] is None


# ---- synthetic / sample data --------------------------------------------- #
def test_synthetic_data_is_deterministic():
    a = generate_synthetic_weather(periods=50, seed=1)
    b = generate_synthetic_weather(periods=50, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_sample_csv_is_valid_and_clean():
    clean, report = clean_weather_data(load_csv(SAMPLE))
    assert len(clean) == 1096
    assert report.warnings == []
    assert {"temperature", "humidity", "rainfall", "wind_speed", "pressure"} <= set(clean.columns)
