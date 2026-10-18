"""Feature engineering: calendar fields, seasons, rolling stats, climatology baseline."""
import numpy as np
import pandas as pd
from .config import BASELINE_END_YEAR

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}


def _climatology(base: pd.DataFrame, col: str) -> dict:
    """Smoothed day-of-year 'normal' per city (31-day circular moving average)."""
    out = {}
    for city, g in base.groupby("city"):
        arr = g.groupby("doy")[col].mean().reindex(range(1, 367)).interpolate(limit_direction="both").to_numpy()
        ext = np.r_[arr[-15:], arr, arr[:15]]
        out[city] = np.convolve(ext, np.ones(31) / 31, mode="same")[15:-15]
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"], df["month"], df["doy"] = df.date.dt.year, df.date.dt.month, df.date.dt.dayofyear
    df["season"] = df["month"].map(SEASONS)
    df["temp_roll30"] = df.groupby("city")["temperature_c"].transform(lambda s: s.rolling(30, min_periods=10).mean())
    df["rain_roll30"] = df.groupby("city")["rainfall_mm"].transform(lambda s: s.rolling(30, min_periods=10).sum())
    clim = _climatology(df[df.year <= BASELINE_END_YEAR], "temperature_c")
    df["temp_clim"] = [clim[c][d - 1] for c, d in zip(df.city, df.doy)]
    return df
