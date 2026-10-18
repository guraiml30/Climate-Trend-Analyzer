"""Virtual simulation: builds a realistic daily climate dataset (no real data access needed)."""
import numpy as np
import pandas as pd
from .config import CITIES, END, SEED, START


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    t_years = (dates - dates[0]).days.to_numpy() / 365.25
    doy = dates.dayofyear.to_numpy()
    year_idx = (dates.year - dates.year[0]).to_numpy()
    frames = []

    for city, p in CITIES.items():
        # Temperature = base + warming trend + seasonal cycle + daily noise
        season = np.sin(2 * np.pi * (doy - 60) / 365.25)  # peaks ~end of May
        temp = p["base"] + p["trend"] * t_years + p["amp"] * season + rng.normal(0, 1.5, n)

        # Rainfall: monsoon-weighted wet-day probability, gamma-distributed amounts
        monsoon = np.exp(-((doy - 215) / 45) ** 2)
        wet = rng.random(n) < (0.05 + 0.6 * monsoon)
        year_factor = rng.lognormal(0, 0.15, year_idx.max() + 1)[year_idx]
        mean_amt = p["rain"] / 66  # ~66 wet days per year
        rain = wet * rng.gamma(2.0, mean_amt / 2, n) * year_factor * (1 + p["rain_trend"] * t_years)

        # Humidity rises with monsoon, falls with heat
        hum = np.clip(p["humid"] + 25 * monsoon - 0.8 * (temp - temp.mean()) + rng.normal(0, 5, n), 10, 100)

        # Inject extreme events (ground truth for evaluating the detector)
        injected = np.zeros(n, dtype=int)
        for _ in range(12):  # heatwaves
            s, length = rng.integers(0, n - 8), rng.integers(3, 8)
            temp[s:s + length] += rng.uniform(5, 8)
            injected[s:s + length] = 1
        for _ in range(10):  # extreme rainfall days
            i = rng.integers(0, n)
            rain[i] = rng.uniform(150, 300)
            injected[i] = 1

        # Real-world data problems: missing values + sensor errors
        for arr in (temp, rain, hum):
            arr[rng.random(n) < 0.015] = np.nan
        for i in rng.integers(0, n, 4):
            temp[i], hum[i] = 99.9, 140.0

        frames.append(pd.DataFrame({
            "date": dates, "city": city,
            "temperature_c": temp.round(2), "rainfall_mm": rain.round(1),
            "humidity_pct": hum.round(1), "injected_event": injected,
        }))
    return pd.concat(frames, ignore_index=True)
