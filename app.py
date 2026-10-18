"""Climate Trend Analyzer API + themed dashboard (FastAPI)."""
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import IMG_DIR, OUT_DIR, PROC_DIR
from src.pipeline import run_pipeline

app = FastAPI(title="🌍 Climate Trend Analyzer API", version="1.0.0",
              description="Trends, anomalies and forecasts from a simulated 30-year climate dataset. "
                          "Open `/` for the interactive dashboard.")
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")


def _csv(name: str, folder: Path = OUT_DIR) -> pd.DataFrame:
    if not (folder / name).exists():
        run_pipeline(verbose=False)
    return pd.read_csv(folder / name)


def _city(df: pd.DataFrame, city):
    if city is None:
        return df
    out = df[df["city"].str.lower() == city.lower()]
    if out.empty:
        raise HTTPException(404, f"Unknown city '{city}'. See /api/cities")
    return out


def _rows(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records"))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "climate-trend-analyzer"}


@app.get("/api/summary")
def summary():
    _csv("annual_summary.csv")
    return json.loads((OUT_DIR / "summary.json").read_text())


@app.get("/api/cities")
def cities():
    return sorted(_csv("annual_summary.csv")["city"].unique().tolist())


@app.get("/api/annual")
def annual(city: str = None):
    return _rows(_city(_csv("annual_summary.csv"), city))


@app.get("/api/trends")
def trends(city: str = None):
    return _rows(_city(_csv("trend_stats.csv"), city))


@app.get("/api/anomalies")
def anomalies(city: str = None, limit: int = Query(20, ge=1, le=500)):
    df = _city(_csv("anomalies.csv"), city).sort_values("date", ascending=False)
    return {"count": int(len(df)), "items": _rows(df.head(limit))}


@app.get("/api/forecast")
def forecast(city: str = None):
    return _rows(_city(_csv("forecast.csv"), city))


@app.get("/api/timeseries")
def timeseries(city: str, year: int = Query(2024, ge=1995, le=2024)):
    df = _city(_csv("climate_features.csv", PROC_DIR), city)
    df = df[df["date"].str.startswith(str(year))]
    return _rows(df[["date", "temperature_c", "rainfall_mm", "humidity_pct", "temp_z"]].round(2))
