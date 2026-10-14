"""AI-Based Weather Prediction System using AI time-series analysis (Streamlit app).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st

try:  # optional: load MONGODB_URI from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from data_processing import (
    COLUMN_LABELS,
    DATE_COLUMN,
    NUMERIC_COLUMNS,
    DataValidationError,
    ValidationReport,
    clean_weather_data,
    compute_overview,
    filter_by_date,
    generate_synthetic_weather,
    load_csv,
    make_regular_series,
)
from database import DB_NAME, COLLECTION_NAME, DatabaseUnavailableError, WeatherDatabase
from forecasting import (
    MAX_HORIZON,
    MIN_OBSERVATIONS,
    ForecastingError,
    actual_vs_predicted_frame,
    future_forecast_frame,
    run_forecast,
)
from visualization import (
    actual_vs_predicted_chart,
    forecast_chart,
    monthly_rainfall_chart,
    rainfall_chart,
    time_series_chart,
    wind_histogram,
)

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "weather.csv"
SRC_SAMPLE = "Sample dataset (synthetic)"
SRC_UPLOAD = "Uploaded CSV"
SRC_MONGO = "MongoDB collection"

st.set_page_config(page_title="AI Weather Prediction System", page_icon="🌦️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1400px;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border-radius: 10px;
                                 padding: 12px 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources and small helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_sample() -> Tuple[pd.DataFrame, ValidationReport]:
    """Load the bundled SYNTHETIC sample dataset (regenerated if the file is missing)."""
    raw = load_csv(SAMPLE_PATH) if SAMPLE_PATH.exists() else generate_synthetic_weather()
    return clean_weather_data(raw)


@st.cache_resource(show_spinner=False)
def get_database() -> WeatherDatabase:
    """Create the (optional) MongoDB connection once; falls back to CSV mode silently."""
    db = WeatherDatabase()
    db.connect()
    return db


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")


def round_numeric(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """Round only numeric columns (leaves date columns untouched)."""
    return df.round({c: digits for c in df.select_dtypes("number").columns})


def fmt(value: Optional[float], unit: str = "", digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:,.{digits}f} {unit}".strip()


def handle_upload(dayfirst: bool) -> None:
    """Read the uploaded CSV, validate it and keep the cleaned result in session state."""
    uploaded = st.session_state.get("csv_uploader")
    if uploaded is None:
        for key in ("upload_df", "upload_report", "upload_name", "upload_error", "upload_sig"):
            st.session_state.pop(key, None)
        return
    signature = (uploaded.name, uploaded.size, dayfirst)
    if st.session_state.get("upload_sig") == signature:
        return
    st.session_state["upload_sig"] = signature
    try:
        raw = load_csv(io.BytesIO(uploaded.getvalue()))
        clean, report = clean_weather_data(raw, dayfirst=dayfirst)
    except DataValidationError as exc:
        st.session_state.update(upload_df=None, upload_report=None, upload_error=str(exc))
        return
    st.session_state.update(upload_df=clean, upload_report=report, upload_name=uploaded.name,
                            upload_error=None, data_source=SRC_UPLOAD)


def resolve_dataset(db: WeatherDatabase) -> Tuple[pd.DataFrame, ValidationReport, str, bool]:
    """Return (dataframe, report, label, is_synthetic) for the selected data source."""
    choice = st.session_state.get("data_source", SRC_SAMPLE)
    if choice == SRC_UPLOAD:
        if st.session_state.get("upload_df") is not None:
            return (st.session_state["upload_df"], st.session_state["upload_report"],
                    f"Uploaded file: {st.session_state.get('upload_name', 'CSV')}", False)
        st.sidebar.info("No valid CSV uploaded yet, so the sample dataset is shown.")
    elif choice == SRC_MONGO and db.is_connected:
        try:
            raw = db.fetch_dataframe()
            if raw.empty:
                st.sidebar.info("The MongoDB collection is empty, so the sample dataset is shown.")
            else:
                clean, report = clean_weather_data(raw)
                return clean, report, f"MongoDB: {DB_NAME}.{COLLECTION_NAME}", False
        except (DatabaseUnavailableError, DataValidationError) as exc:
            st.sidebar.error(f"Could not use MongoDB data: {exc}")
    clean, report = load_sample()
    return clean, report, "Sample dataset (SYNTHETIC demonstration data)", True


# --------------------------------------------------------------------------- #
# Page skeleton: tabs are created first so the upload (Data Management) can
# feed the data source used by every other tab.
# --------------------------------------------------------------------------- #
st.title("🌦️ AI-Based Weather Prediction System")
st.caption("Time-series analysis and ARIMA temperature forecasting for historical weather data.")
tab_overview, tab_hist, tab_forecast, tab_data = st.tabs(
    ["Overview", "Historical Weather Analysis", "AI Forecasting", "Data Management"]
)

db = get_database()

with tab_data:
    st.subheader("Upload weather data")
    st.caption("Required columns: `date`, `temperature`. Optional: `humidity`, `rainfall`, "
               "`wind_speed`, `pressure`. ISO dates (YYYY-MM-DD) are recommended.")
    dayfirst_input = st.checkbox("Dates are written day-first (DD/MM/YYYY)", value=False, key="dayfirst")
    st.file_uploader("Upload a CSV file", type=["csv"], key="csv_uploader")
    handle_upload(dayfirst_input)

# Sidebar
st.sidebar.header("Data source")
options = [SRC_SAMPLE, SRC_UPLOAD] + ([SRC_MONGO] if db.is_connected else [])
if st.session_state.get("data_source") not in options:
    st.session_state["data_source"] = SRC_SAMPLE
st.sidebar.radio("Choose the dataset to analyse", options, key="data_source")
if db.is_connected:
    st.sidebar.success("MongoDB connected")
else:
    st.sidebar.info(f"CSV mode. {db.status_message}")
    if st.sidebar.button("Retry MongoDB connection"):
        get_database.clear()
        st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Forecasts are statistical estimates. Accuracy depends on data quality "
                   "and model performance.")

df, report, dataset_label, is_synthetic = resolve_dataset(db)

if is_synthetic:
    st.warning("⚠️ **Synthetic demonstration data.** The sample dataset was generated by a program "
               "and contains **no actual weather observations**. Upload your own CSV for real analysis.")
st.caption(f"Active dataset: **{dataset_label}** · {len(df):,} records · "
           f"{df[DATE_COLUMN].min():%Y-%m-%d} to {df[DATE_COLUMN].max():%Y-%m-%d}")

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    ov = compute_overview(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total weather records", f"{ov['records']:,}")
    c2.metric("Average temperature", fmt(ov["avg_temperature"], "°C"))
    c3.metric("Average humidity", fmt(ov["avg_humidity"], "%"))
    c4, c5, c6 = st.columns(3)
    c4.metric("Total rainfall", fmt(ov["total_rainfall"], "mm"))
    c5.metric("Average wind speed", fmt(ov["avg_wind_speed"], "km/h"))
    hottest = ov["max_temperature_date"]
    c6.metric("Highest recorded temperature", fmt(ov["max_temperature"], "°C"),
              help=f"Recorded on {hottest:%Y-%m-%d}" if hottest is not None else None)
    st.plotly_chart(time_series_chart(df, "temperature", rolling_window=30), key="overview_temp")
    missing = [COLUMN_LABELS[c][0] for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing:
        st.info("Not available in this dataset: " + ", ".join(missing) + ".")

# --------------------------------------------------------------------------- #
# Historical analysis
# --------------------------------------------------------------------------- #
with tab_hist:
    st.subheader("Historical weather analysis")
    min_d, max_d = df[DATE_COLUMN].min().date(), df[DATE_COLUMN].max().date()
    selected = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                             key="hist_range")
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        st.info("Select both a start and an end date.")
    else:
        view = filter_by_date(df, selected[0], selected[1])
        if view.empty:
            st.warning("No records in the selected date range.")
        else:
            st.caption(f"{len(view):,} records selected.")
            smooth = st.checkbox("Show 7-point rolling mean", value=True, key="hist_smooth")
            window = 7 if smooth else None
            left, right = st.columns(2)
            with left:
                st.plotly_chart(time_series_chart(view, "temperature", window), key="hist_temp")
                st.plotly_chart(rainfall_chart(view), key="hist_rain")
            with right:
                st.plotly_chart(time_series_chart(view, "humidity", window), key="hist_hum")
                st.plotly_chart(time_series_chart(view, "wind_speed", window), key="hist_wind")
            left2, right2 = st.columns(2)
            with left2:
                st.plotly_chart(monthly_rainfall_chart(view), key="hist_rain_month")
            with right2:
                st.plotly_chart(wind_histogram(view), key="hist_wind_hist")
            if "pressure" in view.columns:
                st.plotly_chart(time_series_chart(view, "pressure", window), key="hist_pressure")
            st.markdown("**Descriptive statistics**")
            cols = [c for c in NUMERIC_COLUMNS if c in view.columns]
            st.dataframe(view[cols].describe().T.round(2))

# --------------------------------------------------------------------------- #
# AI forecasting
# --------------------------------------------------------------------------- #
with tab_forecast:
    st.subheader("AI forecasting with ARIMA")
    st.info("The ARIMA model is implemented for **temperature only**. Humidity, rainfall, wind speed "
            "and pressure are analysed historically but are **not** forecast, because they have no "
            "forecasting model in this project.")
    st.selectbox("Weather variable", ["Temperature (°C)"], key="fc_variable")

    k1, k2, k3 = st.columns(3)
    horizon = k1.slider("Forecast horizon (steps ahead, 1 step = 1 record interval)", 1, 90, 14,
                        key="fc_horizon")
    test_pct = k2.slider("Test set size (% of most recent data)", 10, 40, 20, key="fc_test_pct")
    conf_pct = k3.selectbox("Prediction interval", [80, 90, 95], index=2, key="fc_conf",
                            format_func=lambda v: f"{v}%")
    mode = st.radio("ARIMA order", ["Automatic (lowest AIC)", "Manual"], horizontal=True, key="fc_mode")
    manual_order: Optional[Tuple[int, int, int]] = None
    if mode == "Manual":
        o1, o2, o3 = st.columns(3)
        manual_order = (o1.number_input("p (AR)", 0, 10, 1, key="fc_p"),
                        o2.number_input("d (differencing)", 0, 2, 1, key="fc_d"),
                        o3.number_input("q (MA)", 0, 10, 1, key="fc_q"))

    enough = len(df) >= MIN_OBSERVATIONS
    if not enough:
        st.warning(f"Insufficient data: at least {MIN_OBSERVATIONS} records are needed for reliable "
                   f"ARIMA forecasting, but this dataset has {len(df)}.")
    if st.button("Run forecast", type="primary", disabled=not enough, key="fc_run"):
        with st.spinner("Preparing data and fitting the ARIMA model..."):
            try:
                series, info = make_regular_series(df, "temperature")
                result = run_forecast(series, horizon=int(horizon), test_fraction=test_pct / 100.0,
                                      order=manual_order, confidence_level=conf_pct / 100.0)
                st.session_state["forecast"] = {"result": result, "info": info,
                                                "label": dataset_label, "synthetic": is_synthetic}
            except (ForecastingError, DataValidationError) as exc:
                st.session_state.pop("forecast", None)
                st.error(f"Forecast could not be produced: {exc}")

    stored = st.session_state.get("forecast")
    if stored:
        res = stored["result"]
        if stored["label"] != dataset_label:
            st.warning(f"These results were produced from a different dataset ({stored['label']}). "
                       "Run the forecast again for the active dataset.")
        if stored["synthetic"]:
            st.warning("Results below are based on SYNTHETIC demonstration data.")
        if stored["info"]["gaps_filled"]:
            st.caption(f"{stored['info']['gaps_filled']} calendar gap(s) were filled by interpolation "
                       "before modelling.")
        st.markdown(f"**Model:** ARIMA{res.order} · AIC {res.aic:,.1f} · trained on {len(res.train):,} "
                    f"records, tested on the most recent {len(res.test):,}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE (test)", f"{res.metrics['mae']:.2f} °C")
        m2.metric("RMSE (test)", f"{res.metrics['rmse']:.2f} °C")
        m3.metric("Baseline MAE", f"{res.baseline_metrics['mae']:.2f} °C",
                  help="Naive baseline: repeat the last training value over the test period.")
        m4.metric("Baseline RMSE", f"{res.baseline_metrics['rmse']:.2f} °C")
        for note in res.warnings:
            st.warning(note)
        st.plotly_chart(actual_vs_predicted_chart(res), key="fc_avp")
        st.plotly_chart(forecast_chart(res), key="fc_future")
        future_df = future_forecast_frame(res)
        compare_df = actual_vs_predicted_frame(res)
        with st.expander("Forecast values"):
            st.dataframe(round_numeric(future_df))
        with st.expander("Actual vs predicted (test period)"):
            st.dataframe(round_numeric(compare_df))
        st.caption("Forecasts are estimates, not guarantees. Accuracy depends on data quality and "
                   "model performance; this academic project has not been validated for operational use.")

# --------------------------------------------------------------------------- #
# Data management (second pass: validation, preview, downloads, MongoDB)
# --------------------------------------------------------------------------- #
with tab_data:
    if st.session_state.get("upload_error"):
        st.error(f"Upload rejected: {st.session_state['upload_error']}")
    st.subheader("Validation summary")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Rows read", f"{report.rows_in:,}")
    v2.metric("Rows kept", f"{report.rows_out:,}")
    v3.metric("Invalid dates removed", f"{report.invalid_dates_removed:,}")
    v4.metric("Duplicate dates removed", f"{report.duplicate_dates_removed:,}")
    if report.warnings:
        with st.expander(f"Data validation warnings ({len(report.warnings)})", expanded=True):
            for message in report.warnings:
                st.warning(message)
    else:
        st.success("No validation issues found.")

    st.subheader("Preview")
    st.dataframe(df.head(100))
    d1, d2 = st.columns(2)
    d1.download_button("Download cleaned data (CSV)", to_csv_bytes(df), "cleaned_weather_data.csv",
                       "text/csv", key="dl_clean")
    stored = st.session_state.get("forecast")
    if stored:
        d2.download_button("Download forecast (CSV)", to_csv_bytes(future_forecast_frame(stored["result"])),
                           "temperature_forecast.csv", "text/csv", key="dl_forecast")
    else:
        d2.caption("Run a forecast to enable the forecast download.")

    st.subheader("MongoDB")
    if not db.is_connected:
        st.info(f"{db.status_message} To enable MongoDB, copy `.env.example` to `.env`, set "
                "`MONGODB_URI`, and restart the app. Everything else works without it.")
    else:
        st.success(f"Connected to `{DB_NAME}.{COLLECTION_NAME}` with {db.count_records():,} records.")
        if st.button("Save active dataset to MongoDB (upsert by date)", key="db_save"):
            try:
                summary = db.insert_many_records(df)
                st.success(f"Inserted {summary['inserted']}, updated {summary['updated']}, "
                           f"skipped {summary['skipped']}.")
            except (DatabaseUnavailableError, ValueError) as exc:
                st.error(str(exc))
        try:
            a1, a2 = st.columns(2)
            avg = db.get_average_temperature()
            a1.metric("Average temperature (DB)", fmt(avg, "°C"))
            a2.metric("Total rainfall (DB)", fmt(db.get_total_rainfall(), "mm"))
            groups = db.group_by_humidity_level()
            if groups:
                st.markdown("**Records grouped by humidity level** (Low < 40 %, Medium 40–70 %, High ≥ 70 %)")
                st.dataframe(pd.DataFrame(groups).round(2))
        except (DatabaseUnavailableError, ValueError) as exc:
            st.error(str(exc))

        with st.expander("Query records"):
            q1, q2, q3 = st.columns(3)
            min_t = q1.number_input("Min temperature (°C)", value=None, key="q_min_t")
            max_t = q1.number_input("Max temperature (°C)", value=None, key="q_max_t")
            min_h = q2.number_input("Min humidity (%)", value=None, key="q_min_h")
            max_h = q2.number_input("Max humidity (%)", value=None, key="q_max_h")
            min_r = q3.number_input("Min rainfall (mm)", value=None, key="q_min_r")
            max_r = q3.number_input("Max rainfall (mm)", value=None, key="q_max_r")
            q4, q5, q6, q7 = st.columns(4)
            start = q4.date_input("From date", value=None, key="q_start")
            end = q5.date_input("To date", value=None, key="q_end")
            sort_by = q6.selectbox("Sort by", ["date", "temperature", "humidity", "rainfall"], key="q_sort")
            ascending = q7.checkbox("Ascending", value=True, key="q_asc")
            if st.button("Run query", key="q_run"):
                try:
                    rows = db.find_records(min_t, max_t, min_h, max_h, min_r, max_r, start, end,
                                           sort_by=sort_by, ascending=ascending, limit=500)
                    st.caption(f"{len(rows)} record(s) (max 500 shown).")
                    st.dataframe(pd.DataFrame(rows))
                except (DatabaseUnavailableError, ValueError) as exc:
                    st.error(str(exc))

        with st.expander("Update or delete a record"):
            target = st.date_input("Record date", value=min_d, key="u_date")
            field_name = st.selectbox("Field to update", NUMERIC_COLUMNS, key="u_field")
            new_value = st.number_input("New value", value=0.0, key="u_value")
            u1, u2 = st.columns(2)
            if u1.button("Update record", key="u_btn"):
                try:
                    ok = db.update_record(target, {field_name: new_value})
                    if ok:
                        st.success("Record updated.")
                    else:
                        st.warning("No record exists for that date.")
                except (DatabaseUnavailableError, ValueError) as exc:
                    st.error(str(exc))
            confirm = u2.checkbox("Confirm deletion", key="d_confirm")
            if u2.button("Delete record", key="d_btn", disabled=not confirm):
                try:
                    ok = db.delete_record(target)
                    if ok:
                        st.success("Record deleted.")
                    else:
                        st.warning("No record exists for that date.")
                except (DatabaseUnavailableError, ValueError) as exc:
                    st.error(str(exc))
