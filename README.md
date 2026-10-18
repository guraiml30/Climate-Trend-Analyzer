# 🌍 Climate Trend Analyzer

End-to-end climate analytics project: simulated 30-year daily weather data → cleaning → trend analysis → anomaly detection → 5-year forecast → charts, report and a **themed REST API + dashboard**, all launched with one command: `python runall.py`.

## Problem statement
Governments, city planners and researchers need to know *how fast* a region is warming, how rainfall is shifting, and *when* extreme events occur. This project automates that workflow on a reproducible, simulated dataset (no institutional data access needed).

## Industry relevance
Climate-risk teams, smart-city planners, agriculture/insurance analysts and environmental agencies all run trend + anomaly + forecast pipelines like this one.

## Tech stack
Python · Pandas · NumPy · Matplotlib · Statsmodels (OLS, Holt-Winters) · Scikit-learn (IsolationForest) · FastAPI · Uvicorn · Chart.js · Pytest · Docker

## Architecture
```
 Simulator ─► Cleaning ─► Features ─► Trend (OLS) ─┐
 (data_generator) (preprocess) (features)  (trend)  │
                                  ├─► Anomaly (z-score, heatwave, IsolationForest) ─┤
                                  └─► Forecast (Holt-Winters) ──────────────────────┤
                                                                                    ▼
                          Charts (images/) · CSVs (outputs/) · Report (reports/) · FastAPI dashboard
```

## Folder structure
```
climate-trend-analyzer/
├── runall.py            # one command: pipeline + API
├── src/                 # config, data_generator, preprocess, features, trend, anomaly, forecast, visualize, report, pipeline
├── api/                 # app.py (FastAPI) + dashboard.html (themed UI)
├── tests/               # pytest checks
├── data/raw|processed/  # generated datasets
├── outputs/             # result CSVs + summary.json
├── images/              # charts for README/proof
├── reports/             # insights_report.md
├── docs/GUIDE.md        # GitHub plan, resume, interview Q&A, troubleshooting
├── Dockerfile · requirements.txt · .gitignore
```

## Installation
**Windows**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
**Mac/Linux**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Check: `python -c "import pandas, statsmodels, fastapi; print('OK')"` (Python 3.10+ recommended).

## How to run
| Command | What it does |
|---|---|
| `python runall.py` | Full pipeline, then API at http://127.0.0.1:8000 |
| `python runall.py --open` | Same, and opens the dashboard in your browser |
| `python runall.py --no-api` | Analysis only |
| `python runall.py --skip-pipeline` | API only (reuse existing outputs) |
| `python runall.py --host 0.0.0.0 --port 9000` | Custom host/port |
| `pytest -q` | Run tests |

**API endpoints:** `/` dashboard · `/docs` Swagger · `/api/summary` · `/api/cities` · `/api/annual?city=Delhi` · `/api/trends` · `/api/anomalies?city=&limit=` · `/api/forecast` · `/api/timeseries?city=Delhi&year=2024`

**Deploy:** `docker build -t climate-api . && docker run -p 8000:8000 climate-api`, or on Render/Railway use start command `python runall.py --host 0.0.0.0 --port $PORT`.

## Dataset & simulation workflow
Daily temperature, rainfall and humidity for Delhi, Mumbai and Bengaluru (1995–2024) = base climate + warming trend + seasonal cycle + noise, monsoon-weighted gamma rainfall, plus injected heatwaves, extreme-rain days, ~1.5% missing values and sensor errors. Injected events act as ground truth to measure detector recall. To use real data, replace `generate()` with a CSV loader (NASA POWER, NOAA, Kaggle) using the same columns.

## Results (fill in after running)
| Metric | Value |
|---|---|
| Warming per decade (Delhi/Mumbai/Bengaluru) | see `outputs/trend_stats.csv` |
| Detector recall on injected events | see `outputs/summary.json` |
| Forecast MAE (24-month holdout) | see `outputs/forecast_metrics.csv` |

## Screenshots
![Temperature trend](images/01_temperature_trend.png)
![Rainfall trend](images/02_rainfall_trend.png)
![Seasonal pattern](images/03_seasonal_pattern.png)
![Anomalies](images/04_anomaly_detection.png)
![Heatmap](images/05_temp_anomaly_heatmap.png)
![Forecast](images/06_forecast.png)
![Dashboard](images/07_dashboard.png)

## Future improvements
Real NASA/NOAA data · more cities & geospatial map · Prophet/SARIMA · pollution-climate correlation · live weather API · climate risk score · automated PDF reports.

## Learning outcomes
Time-series analysis, regression with significance testing, anomaly detection (statistical + ML), forecasting validation, REST API design, reproducible pipelines.

## Author
Your Name · [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
