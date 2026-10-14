"""Regenerate ``sample_data/weather.csv`` (SYNTHETIC demonstration data)."""
from pathlib import Path

from data_processing import generate_synthetic_weather

if __name__ == "__main__":
    out = Path(__file__).parent / "sample_data" / "weather.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_weather()
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(df)} synthetic rows to {out}")
