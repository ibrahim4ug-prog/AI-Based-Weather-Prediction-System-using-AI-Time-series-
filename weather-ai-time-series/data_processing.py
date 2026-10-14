"""Loading, validating and cleaning historical weather data.

All demonstration data produced by :func:`generate_synthetic_weather` is
SYNTHETIC. It is not an actual weather observation.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

DATE_COLUMN = "date"
TEMPERATURE_COLUMN = "temperature"
OPTIONAL_COLUMNS: List[str] = ["humidity", "rainfall", "wind_speed", "pressure"]
NUMERIC_COLUMNS: List[str] = [TEMPERATURE_COLUMN] + OPTIONAL_COLUMNS
ALL_COLUMNS: List[str] = [DATE_COLUMN] + NUMERIC_COLUMNS

#: Human readable label and unit for every column (used by charts and tables).
COLUMN_LABELS: Dict[str, Tuple[str, str]] = {
    "temperature": ("Temperature", "°C"),
    "humidity": ("Relative humidity", "%"),
    "rainfall": ("Rainfall", "mm"),
    "wind_speed": ("Wind speed", "km/h"),
    "pressure": ("Atmospheric pressure", "hPa"),
}

#: Physically plausible ranges; values outside are treated as invalid.
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature": (-90.0, 60.0),
    "humidity": (0.0, 100.0),
    "rainfall": (0.0, 2000.0),
    "wind_speed": (0.0, 500.0),
    "pressure": (300.0, 1100.0),
}

COLUMN_ALIASES: Dict[str, str] = {
    **{k: "date" for k in ("date", "datetime", "timestamp", "time", "day", "observation_date")},
    **{k: "temperature" for k in ("temperature", "temp", "avg_temp", "temperature_c",
                                  "mean_temperature", "tavg", "air_temperature")},
    **{k: "humidity" for k in ("humidity", "relative_humidity", "rh", "humidity_percent")},
    **{k: "rainfall" for k in ("rainfall", "rain", "precipitation", "precip", "prcp", "rainfall_mm")},
    **{k: "wind_speed" for k in ("wind_speed", "windspeed", "wind", "wspd", "wind_speed_kmh")},
    **{k: "pressure" for k in ("pressure", "atmospheric_pressure", "air_pressure", "pres", "pressure_hpa")},
}

MIN_YEAR, MAX_YEAR = 1900, 2100
HIGH_MISSING_FRACTION = 0.20


class DataValidationError(ValueError):
    """Raised when an uploaded dataset cannot be used at all."""


@dataclass
class ValidationReport:
    """Summary of what validation/cleaning found and changed."""

    rows_in: int = 0
    rows_out: int = 0
    invalid_dates_removed: int = 0
    duplicate_dates_removed: int = 0
    invalid_numeric_values: int = 0
    out_of_range_values: int = 0
    missing_values_filled: Dict[str, int] = field(default_factory=dict)
    input_was_unsorted: bool = False
    columns_present: List[str] = field(default_factory=list)
    columns_missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_csv(source: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    """Read a CSV file (path or file-like object) into a raw DataFrame.

    Raises:
        DataValidationError: if the file is missing, empty or not parseable.
    """
    try:
        df = pd.read_csv(source)
    except FileNotFoundError as exc:
        raise DataValidationError(f"File not found: {source}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not parse the file as CSV: {exc}") from exc
    if df.empty:
        raise DataValidationError("The CSV file has a header but contains no data rows.")
    return df


def _normalise_name(name: Any) -> str:
    text = re.sub(r"\(.*?\)", "", str(name).lower())
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalise_columns(df: pd.DataFrame, report: Optional[ValidationReport] = None) -> pd.DataFrame:
    """Rename columns to canonical names using :data:`COLUMN_ALIASES`.

    Unknown columns are dropped (with a warning); duplicate aliases keep the first.
    """
    rename: Dict[str, str] = {}
    seen: Dict[str, str] = {}
    ignored: List[str] = []
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(_normalise_name(col))
        if canonical is None:
            ignored.append(str(col))
        elif canonical in seen:
            ignored.append(str(col))
            if report is not None:
                report.add_warning(
                    f"Column '{col}' duplicates '{seen[canonical]}' (both map to '{canonical}'); "
                    "the first one was used."
                )
        else:
            seen[canonical] = str(col)
            rename[col] = canonical
    if ignored and report is not None:
        report.add_warning("Ignored unrecognised columns: " + ", ".join(sorted(set(ignored))) + ".")
    return df[list(rename.keys())].rename(columns=rename)


def _parse_dates(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Parse mixed date strings to timezone-naive timestamps (invalid -> NaT)."""
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="mixed", errors="coerce", utc=True, dayfirst=dayfirst)
    return parsed.dt.tz_localize(None)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_weather_data(
    df: pd.DataFrame, dayfirst: bool = False
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a raw weather DataFrame.

    Steps: rename columns -> parse dates (drop invalid) -> coerce numbers
    (invalid/out-of-range become missing) -> sort chronologically -> drop
    duplicate dates (last occurrence kept) -> fill missing values.

    Missing values: linear interpolation (then edge fill) for temperature,
    humidity, wind speed and pressure; missing rainfall is assumed to be 0 mm
    (an assumption, reported as a warning). Columns that are entirely missing
    are dropped.

    Returns:
        ``(clean_df, report)`` where ``clean_df`` has a ``date`` column and
        float columns, sorted by date with a fresh integer index.

    Raises:
        DataValidationError: if required columns are missing or no usable rows remain.
    """
    report = ValidationReport(rows_in=int(len(df)))
    if df is None or df.empty:
        raise DataValidationError("The dataset is empty.")

    work = normalise_columns(df, report)
    missing_required = [c for c in (DATE_COLUMN, TEMPERATURE_COLUMN) if c not in work.columns]
    if missing_required:
        raise DataValidationError(
            "Missing required column(s): " + ", ".join(missing_required)
            + f". Columns found: {', '.join(map(str, df.columns))}. "
            "Expected at least 'date' and 'temperature'."
        )

    # Dates
    work = work.copy()
    work[DATE_COLUMN] = _parse_dates(work[DATE_COLUMN], dayfirst=dayfirst)
    year = work[DATE_COLUMN].dt.year
    valid_date = work[DATE_COLUMN].notna() & year.between(MIN_YEAR, MAX_YEAR)
    report.invalid_dates_removed = int((~valid_date).sum())
    if report.invalid_dates_removed:
        report.add_warning(f"Removed {report.invalid_dates_removed} row(s) with missing or invalid dates.")
    work = work.loc[valid_date].copy()
    if work.empty:
        raise DataValidationError("No rows with valid dates remain after validation.")
    report.input_was_unsorted = not work[DATE_COLUMN].is_monotonic_increasing

    # Numbers
    for col in NUMERIC_COLUMNS:
        if col not in work.columns:
            continue
        original = work[col]
        numeric = pd.to_numeric(original, errors="coerce").astype(float)
        numeric = numeric.where(np.isfinite(numeric))
        bad = int((original.notna() & numeric.isna()).sum())
        low, high = VALID_RANGES[col]
        out_of_range = numeric.notna() & ((numeric < low) | (numeric > high))
        n_range = int(out_of_range.sum())
        numeric = numeric.mask(out_of_range)
        report.invalid_numeric_values += bad
        report.out_of_range_values += n_range
        if bad:
            report.add_warning(f"'{col}': {bad} non-numeric value(s) treated as missing.")
        if n_range:
            report.add_warning(
                f"'{col}': {n_range} value(s) outside the plausible range {low:g} to {high:g} "
                f"{COLUMN_LABELS[col][1]} treated as missing."
            )
        work[col] = numeric

    # Chronological order + duplicates (stable sort keeps file order for equal dates)
    if report.input_was_unsorted:
        report.add_warning("Records were not in chronological order and have been sorted by date.")
    work = work.sort_values(DATE_COLUMN, kind="mergesort")
    dup = work[DATE_COLUMN].duplicated(keep="last")
    report.duplicate_dates_removed = int(dup.sum())
    if report.duplicate_dates_removed:
        report.add_warning(
            f"Removed {report.duplicate_dates_removed} duplicate date record(s); "
            "the last occurrence of each date was kept."
        )
    work = work.loc[~dup].reset_index(drop=True)

    # Drop entirely-empty columns, then fill missing values
    for col in [c for c in NUMERIC_COLUMNS if c in work.columns]:
        if work[col].isna().all():
            if col == TEMPERATURE_COLUMN:
                raise DataValidationError("The 'temperature' column has no valid numeric values.")
            work = work.drop(columns=[col])
            report.add_warning(f"Column '{col}' has no valid values and was dropped.")
            continue
        n_missing = int(work[col].isna().sum())
        if n_missing == 0:
            continue
        report.missing_values_filled[col] = n_missing
        if n_missing / len(work) > HIGH_MISSING_FRACTION:
            report.add_warning(
                f"'{col}': {n_missing} of {len(work)} values ({n_missing / len(work):.0%}) were missing; "
                "results based on this column may be unreliable."
            )
        if col == "rainfall":
            work[col] = work[col].fillna(0.0)
            report.add_warning(
                f"'rainfall': {n_missing} missing value(s) were assumed to be 0 mm."
            )
        else:
            work[col] = work[col].interpolate(method="linear", limit_direction="both")
            report.add_warning(f"'{col}': {n_missing} missing value(s) filled by linear interpolation.")

    present = [c for c in ALL_COLUMNS if c in work.columns]
    work = work[present]
    report.rows_out = int(len(work))
    report.columns_present = present
    report.columns_missing = [c for c in ALL_COLUMNS if c not in present]
    if work.empty:
        raise DataValidationError("No usable rows remain after validation.")
    return work, report


# --------------------------------------------------------------------------- #
# Helpers for analysis
# --------------------------------------------------------------------------- #
def filter_by_date(
    df: pd.DataFrame,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Return rows with ``start <= date <= end`` (inclusive, whole end day)."""
    out = df
    if start is not None:
        out = out[out[DATE_COLUMN] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out[DATE_COLUMN] < pd.Timestamp(end).normalize() + pd.Timedelta(days=1)]
    return out.reset_index(drop=True)


def compute_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Headline statistics; entries are ``None`` when the column is unavailable."""
    def col_stat(col: str, func: str) -> Optional[float]:
        if col not in df.columns or df.empty:
            return None
        return float(getattr(df[col], func)())

    max_temp = col_stat(TEMPERATURE_COLUMN, "max")
    max_temp_date = None
    if max_temp is not None:
        max_temp_date = df.loc[df[TEMPERATURE_COLUMN].idxmax(), DATE_COLUMN]
    return {
        "records": int(len(df)),
        "avg_temperature": col_stat("temperature", "mean"),
        "avg_humidity": col_stat("humidity", "mean"),
        "total_rainfall": col_stat("rainfall", "sum"),
        "avg_wind_speed": col_stat("wind_speed", "mean"),
        "max_temperature": max_temp,
        "max_temperature_date": max_temp_date,
    }


def infer_frequency(index: pd.DatetimeIndex) -> Any:
    """Infer the sampling frequency (as a resample rule) from the most common time step.

    The most common step (smallest one on ties) is used instead of the median so that
    calendar gaps in the data do not distort the inferred spacing.
    """
    if len(index) < 2:
        return "D"
    steps = pd.Series(index).diff().dropna()
    steps = steps[steps > pd.Timedelta(0)]
    if steps.empty:
        return "D"
    counts = steps.value_counts()
    delta = counts[counts == counts.max()].index.min()
    if delta == pd.Timedelta(days=1):
        return "D"
    if delta == pd.Timedelta(hours=1):
        return "h"
    return pd.tseries.frequencies.to_offset(delta)


def make_regular_series(
    df: pd.DataFrame, column: str = TEMPERATURE_COLUMN, freq: Any = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Turn a cleaned DataFrame column into an evenly spaced series for ARIMA.

    Gaps in the calendar are filled by linear interpolation (reported in the
    returned info dict). The result has a DatetimeIndex with ``freq`` set.
    """
    if df is None or df.empty:
        raise DataValidationError("The dataset is empty.")
    if column not in df.columns or DATE_COLUMN not in df.columns:
        raise DataValidationError(f"Column '{column}' is not available in the dataset.")
    s = df.set_index(DATE_COLUMN)[column].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    rule = infer_frequency(s.index) if freq is None else freq
    regular = s.resample(rule).mean()
    gaps = int(regular.isna().sum())
    regular = regular.interpolate(method="linear", limit_direction="both")
    regular.name = column
    info = {
        "frequency": "daily" if rule == "D" else ("hourly" if rule == "h" else str(rule)),
        "observations": int(len(regular)),
        "gaps_filled": gaps,
    }
    return regular, info


# --------------------------------------------------------------------------- #
# Synthetic demonstration data
# --------------------------------------------------------------------------- #
def generate_synthetic_weather(
    start: str = "2022-01-01", periods: int = 1096, seed: int = 42
) -> pd.DataFrame:
    """Generate deterministic SYNTHETIC daily weather (not real observations).

    Seasonal sinusoids plus random noise; values are plausible-looking but
    carry no real-world meaning.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (doy - 105) / 365.25)

    noise = np.zeros(periods)
    eps = rng.normal(0.0, 1.6, periods)
    for i in range(1, periods):
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    temperature = 24.0 + 9.0 * season + noise

    wet_season = np.clip(np.sin(2 * np.pi * (doy - 150) / 365.25), 0.0, None)
    humidity = np.clip(58.0 - 0.8 * (temperature - 24.0) + 25.0 * wet_season
                       + rng.normal(0.0, 6.0, periods), 15.0, 100.0)
    rain_today = rng.random(periods) < (0.06 + 0.45 * wet_season)
    rainfall = np.where(rain_today, rng.gamma(1.5, 7.0, periods), 0.0)
    wind = np.clip(rng.gamma(4.0, 2.6, periods) + 2.0 * wet_season, 0.0, 80.0)

    p_noise = np.zeros(periods)
    p_eps = rng.normal(0.0, 1.0, periods)
    for i in range(1, periods):
        p_noise[i] = 0.9 * p_noise[i - 1] + p_eps[i]
    pressure = 1008.0 - 5.0 * season + p_noise

    return pd.DataFrame(
        {
            "date": dates,
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "rainfall": np.round(rainfall, 1),
            "wind_speed": np.round(wind, 1),
            "pressure": np.round(pressure, 1),
        }
    )
