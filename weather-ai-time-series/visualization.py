"""Interactive Plotly charts. Every chart has a title, axis labels and units."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from data_processing import COLUMN_LABELS, DATE_COLUMN
from forecasting import ForecastResult

COLORS = {
    "temperature": "#e4572e",
    "humidity": "#2e86de",
    "rainfall": "#1b6ca8",
    "wind_speed": "#2a9d8f",
    "pressure": "#7b5ea7",
    "actual": "#2d3142",
    "predicted": "#e4572e",
    "band": "rgba(228, 87, 46, 0.18)",
}


def _style(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=380,
    )
    return fig


def empty_figure(message: str = "No data available") -> go.Figure:
    """Placeholder chart shown when there is nothing to plot."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _has(df: Optional[pd.DataFrame], column: str) -> bool:
    return df is not None and not df.empty and column in df.columns and DATE_COLUMN in df.columns


def time_series_chart(df: pd.DataFrame, column: str, rolling_window: Optional[int] = 7) -> go.Figure:
    """Line chart of one column over time, with an optional rolling mean overlay."""
    if not _has(df, column):
        return empty_figure(f"No '{column}' data available")
    label, unit = COLUMN_LABELS[column]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[DATE_COLUMN], y=df[column], mode="lines", name=f"Daily {label.lower()}",
                             line=dict(color=COLORS[column], width=1.2), opacity=0.55 if rolling_window else 1.0))
    if rolling_window and rolling_window > 1 and len(df) >= rolling_window:
        fig.add_trace(go.Scatter(x=df[DATE_COLUMN], y=df[column].rolling(rolling_window).mean(), mode="lines",
                                 name=f"{rolling_window}-point rolling mean",
                                 line=dict(color=COLORS["actual"], width=2.2)))
    return _style(fig, f"{label} over time", "Date", f"{label} ({unit})")


def rainfall_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of rainfall per record."""
    if not _has(df, "rainfall"):
        return empty_figure("No rainfall data available")
    fig = go.Figure(go.Bar(x=df[DATE_COLUMN], y=df["rainfall"], name="Rainfall",
                           marker_color=COLORS["rainfall"]))
    return _style(fig, "Rainfall over time", "Date", "Rainfall (mm)")


def monthly_rainfall_chart(df: pd.DataFrame) -> go.Figure:
    """Total rainfall per calendar month."""
    if not _has(df, "rainfall"):
        return empty_figure("No rainfall data available")
    monthly = df.set_index(DATE_COLUMN)["rainfall"].resample("MS").sum()
    fig = go.Figure(go.Bar(x=monthly.index, y=monthly.values, name="Monthly total",
                           marker_color=COLORS["rainfall"]))
    return _style(fig, "Monthly rainfall totals", "Month", "Total rainfall (mm)")


def wind_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of wind speeds."""
    if not _has(df, "wind_speed"):
        return empty_figure("No wind-speed data available")
    fig = go.Figure(go.Histogram(x=df["wind_speed"], nbinsx=30, name="Wind speed",
                                 marker_color=COLORS["wind_speed"]))
    fig.update_layout(hovermode="closest")
    return _style(fig, "Wind-speed distribution", "Wind speed (km/h)", "Number of records").update_layout(
        hovermode="closest")


def _band(fig: go.Figure, frame: pd.DataFrame, name: str) -> None:
    fig.add_trace(go.Scatter(x=frame.index, y=frame["upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=COLORS["band"], name=name, hoverinfo="skip"))


def actual_vs_predicted_chart(result: ForecastResult, context_points: int = 60) -> go.Figure:
    """Training tail, actual test values and ARIMA predictions with the interval band."""
    tail = result.train.iloc[-context_points:]
    pred = result.test_predictions
    pct = f"{result.confidence_level:.0%}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail.index, y=tail.values, mode="lines", name="Training data (tail)",
                             line=dict(color="#9aa0a6", width=1.5)))
    _band(fig, pred, f"{pct} prediction interval")
    fig.add_trace(go.Scatter(x=result.test.index, y=result.test.values, mode="lines", name="Actual (test)",
                             line=dict(color=COLORS["actual"], width=2)))
    fig.add_trace(go.Scatter(x=pred.index, y=pred["forecast"], mode="lines", name="Predicted (ARIMA)",
                             line=dict(color=COLORS["predicted"], width=2, dash="dash")))
    return _style(fig, "Actual vs predicted temperature (test period)", "Date", "Temperature (°C)")


def forecast_chart(result: ForecastResult, history_points: int = 90) -> go.Figure:
    """Recent history followed by the future forecast and its prediction interval."""
    history = pd.concat([result.train, result.test]).iloc[-history_points:]
    future = result.future_forecast
    pct = f"{result.confidence_level:.0%}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Observed",
                             line=dict(color=COLORS["actual"], width=2)))
    _band(fig, future, f"{pct} prediction interval")
    fig.add_trace(go.Scatter(x=future.index, y=future["forecast"], mode="lines+markers",
                             name="Forecast (ARIMA)", line=dict(color=COLORS["predicted"], width=2),
                             marker=dict(size=4)))
    return _style(fig, f"Temperature forecast: next {result.horizon} step(s)", "Date", "Temperature (°C)")
