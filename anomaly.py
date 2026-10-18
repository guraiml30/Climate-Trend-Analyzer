"""Anomaly detection: z-score on de-seasonalised temperature, heatwave runs, extreme rain, IsolationForest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect(df: pd.DataFrame, z_thr: float = 3.0):
    df = df.reset_index(drop=True).copy()
    df["temp_anom"] = df["temperature_c"] - df["temp_clim"]                       # vs 1995-2009 normal
    df["temp_resid"] = df["temp_anom"] - df.groupby(["city", "year"])["temp_anom"].transform("mean")
    df["temp_z"] = df["temp_resid"] / df.groupby("city")["temp_resid"].transform("std")
    df["flag_temp"] = df["temp_z"].abs() >= z_thr

    wet = df["rainfall_mm"] > 1
    p99 = df[wet].groupby("city")["rainfall_mm"].quantile(0.99)
    df["flag_rain"] = wet & (df["rainfall_mm"] > df["city"].map(p99))

    df["flag_heatwave"] = False                                                   # >=3 consecutive hot days
    for _, idx in df.groupby("city").groups.items():
        hot = df.loc[idx, "temp_z"] >= 2
        run = (hot != hot.shift()).cumsum()
        df.loc[idx, "flag_heatwave"] = hot & (hot.groupby(run).transform("size") >= 3)

    iso = IsolationForest(contamination=0.01, random_state=42)                    # ML cross-check
    df["flag_iforest"] = iso.fit_predict(df[["temp_z", "rainfall_mm", "humidity_pct"]].fillna(0)) == -1

    df["anomaly_type"] = np.select([df.flag_rain, df.flag_heatwave, df.flag_temp],
                                   ["Extreme rainfall", "Heatwave", "Temperature anomaly"], default="")
    flagged = df["anomaly_type"] != ""
    anoms = df.loc[flagged, ["date", "city", "anomaly_type", "temperature_c", "rainfall_mm", "temp_z"]].round(2)
    anoms = anoms.rename(columns={"anomaly_type": "type"})

    inj = df["injected_event"] == 1
    hit = int((flagged & inj).sum())
    metrics = dict(recall_injected=round(hit / max(int(inj.sum()), 1), 3),
                   precision_vs_injected=round(hit / max(int(flagged.sum()), 1), 3),
                   iforest_overlap=round(float((df.flag_iforest & flagged).sum() / max(int(df.flag_iforest.sum()), 1)), 3),
                   flagged_days=int(flagged.sum()))
    return df, anoms.reset_index(drop=True), metrics
