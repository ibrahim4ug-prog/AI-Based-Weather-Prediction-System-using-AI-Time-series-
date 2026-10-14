"""End-to-end smoke test of the Streamlit UI using Streamlit's AppTest harness."""
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MONGODB_URI", "")  # force CSV mode
    return AppTest.from_file(str(APP), default_timeout=120)


def test_app_renders_without_exceptions(app):
    at = app.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Overview", "Historical Weather Analysis", "AI Forecasting", "Data Management"]


def test_forecast_button_runs_model(app):
    pytest.importorskip("statsmodels")
    at = app.run()
    at.button(key="fc_run").click().run()
    assert not at.exception
    assert not at.error
    assert "forecast" in at.session_state
