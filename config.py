"""Central configuration: paths, simulation settings and the project colour theme."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "images"
REPORT_DIR = ROOT / "reports"
for _d in (RAW_DIR, PROC_DIR, OUT_DIR, IMG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
START, END = "1995-01-01", "2024-12-31"
BASELINE_END_YEAR = 2009  # climatology "normal" = 1995-2009

# Simulated city climates: base temp (C), seasonal amplitude, warming (C/yr),
# annual rainfall (mm), rainfall trend (fraction/yr), base humidity (%)
CITIES = {
    "Delhi":     dict(base=25.0, amp=10.0, trend=0.040, rain=700,  rain_trend=-0.004, humid=55),
    "Mumbai":    dict(base=27.5, amp=3.5,  trend=0.030, rain=2400, rain_trend=0.006,  humid=75),
    "Bengaluru": dict(base=24.0, amp=3.0,  trend=0.025, rain=900,  rain_trend=0.002,  humid=65),
}

# One colour theme shared by charts, report and the API dashboard
THEME = dict(bg="#0b1d2a", panel="#12293b", grid="#27465c", text="#e6f1f7",
             ocean="#1e88e5", teal="#19c3b1", heat="#ff7043", sand="#f4e3b1")
CITY_COLORS = {"Delhi": THEME["heat"], "Mumbai": THEME["ocean"], "Bengaluru": THEME["teal"]}
