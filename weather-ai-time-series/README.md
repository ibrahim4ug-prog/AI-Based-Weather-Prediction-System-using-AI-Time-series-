# AI-Based Weather Prediction System using AI Time-Series Analysis

An academic project that demonstrates weather data management, MongoDB operations, ARIMA time-series
forecasting, model evaluation and interactive visualisation in a Streamlit dashboard.

> **Important:** `sample_data/weather.csv` is **synthetic** (program-generated). It contains no actual
> weather observations. Forecasts are statistical estimates; accuracy depends on data quality and model
> performance. This project has **not** been validated for operational use.

## Features
- **Data management** – CSV upload, column-name aliases, validation of dates and numbers, plausible-range
  checks, duplicate-date removal (last record kept), chronological sorting, missing-value handling,
  downloadable cleaned data.
- **Forecasting** – ARIMA (statsmodels) for **temperature only**. Chronological train/test split (no
  shuffling), automatic order search by AIC or manual `(p, d, q)`, MAE/RMSE, a naive "repeat last value"
  baseline for honest comparison, prediction intervals, actual-vs-predicted and future-forecast charts.
  Humidity, rainfall, wind speed and pressure are analysed historically but **not** forecast.
- **Dashboard** – Overview, Historical Weather Analysis (date-range filter, descriptive statistics),
  AI Forecasting, Data Management.
- **MongoDB (optional)** – database `weather_prediction`, collection `weather`: insert/upsert, retrieve,
  filter, sort, average temperature, total rainfall, group by humidity level, update, delete. Falls back to
  CSV mode when MongoDB is not configured or unreachable.

## Installation
Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional: MongoDB
```bash
cp .env.example .env             # Windows: copy .env.example .env
# edit .env and set MONGODB_URI, e.g. mongodb://localhost:27017
```
Restart the app. The sidebar shows *MongoDB connected* and the **Data Management** tab offers saving the
active dataset, queries, aggregations, update and delete. The connection string is read only from the
environment and is never logged. `.env` is git-ignored.

## CSV format
| column | required | unit |
|---|---|---|
| `date` | yes | ISO `YYYY-MM-DD` recommended (tick "day-first" for `DD/MM/YYYY`) |
| `temperature` | yes | °C |
| `humidity` | no | % (0–100) |
| `rainfall` | no | mm (≥ 0) |
| `wind_speed` | no | km/h (≥ 0) |
| `pressure` | no | hPa |

Handling rules: invalid dates → row removed; non-numeric or out-of-range numbers → treated as missing;
duplicate dates → last kept; missing values → linear interpolation (rainfall assumed 0 mm); all changes are
listed as warnings in the app. At least 30 records are needed to forecast.

## Tests
```bash
pytest
```
Covers CSV loading/validation, missing values, duplicate dates, chronological splitting, ARIMA training and
forecasting, MAE/RMSE, MongoDB fallback and CRUD (via `mongomock`), plus a Streamlit `AppTest` smoke test.
Tests needing an optional package skip themselves if it is not installed.

## Project structure
```
weather-ai-time-series/
├── app.py                  Streamlit dashboard
├── data_processing.py      loading, cleaning, validation, synthetic data generator
├── forecasting.py          ARIMA training, evaluation, forecasting
├── metrics.py              MAE / RMSE
├── database.py             optional MongoDB integration
├── visualization.py        Plotly charts
├── generate_sample_data.py regenerates sample_data/weather.csv
├── sample_data/            synthetic demo dataset (+ README)
├── tests/                  pytest suite
├── requirements.txt  .env.example  .gitignore  pytest.ini
```

## Known limitations
- Plain ARIMA does **not** model annual seasonality. Over weeks-ahead horizons the forecast flattens and the
  intervals widen. Seasonal ARIMA (SARIMAX) or Fourier terms would be the natural extension.
- Only temperature is forecast; one model, one variable.
- Test-set evaluation is a single multi-step forecast over the most recent data, not rolling-origin
  cross-validation. Compare the reported error with the naive baseline shown next to it.
- Interpolating missing values uses neighbouring records, including some after a gap.
- Dates such as `03/04/2024` are ambiguous; use ISO dates or the day-first option.
