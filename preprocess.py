"""Data cleaning: parse dates, remove impossible values, fill gaps."""
import pandas as pd

VALUE_COLS = ["temperature_c", "rainfall_mm", "humidity_pct"]
VALID = {"temperature_c": (-10, 55), "rainfall_mm": (0, 1000), "humidity_pct": (0, 100)}


def clean(raw: pd.DataFrame):
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(["city", "date"]).sort_values(["city", "date"])

    invalid = 0
    for col, (lo, hi) in VALID.items():  # sensor errors -> NaN
        bad = df[col].notna() & ~df[col].between(lo, hi)
        invalid += int(bad.sum())
        df.loc[bad, col] = float("nan")
    missing = int(df[VALUE_COLS].isna().sum().sum())

    parts = []
    for _, g in df.groupby("city"):
        g = g.set_index("date").copy()
        cols = ["temperature_c", "humidity_pct"]
        g[cols] = g[cols].interpolate(method="time", limit=7, limit_direction="both")
        g["rainfall_mm"] = g["rainfall_mm"].fillna(0)  # never interpolate rain
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out, {"invalid_values_removed": invalid, "missing_values_filled": missing}
