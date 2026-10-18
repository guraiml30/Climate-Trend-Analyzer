"""Charts saved to images/ using the project's ocean/teal/heat theme."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import CITY_COLORS, IMG_DIR, THEME as T


def _style():
    plt.rcParams.update({"figure.facecolor": T["bg"], "axes.facecolor": T["panel"], "axes.edgecolor": T["grid"],
                         "axes.labelcolor": T["text"], "text.color": T["text"], "xtick.color": T["text"],
                         "ytick.color": T["text"], "grid.color": T["grid"], "axes.grid": True, "grid.alpha": .4,
                         "legend.facecolor": T["panel"], "legend.edgecolor": T["grid"], "font.size": 11})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=150)
    plt.close(fig)


def _trend_plot(annual, col, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for city, g in annual.groupby("city"):
        c = CITY_COLORS.get(city)
        ax.plot(g.year, g[col], "o-", color=c, ms=4, lw=1.4, label=city)
        k, b = np.polyfit(g.year, g[col], 1)
        ax.plot(g.year, k * g.year + b, "--", color=c, alpha=.8)
    ax.set(title=title, xlabel="Year", ylabel=ylabel)
    ax.legend()
    _save(fig, name)


def plot_all(df, annual, fc):
    _style()
    _trend_plot(annual, "temp_mean", "Mean temperature (°C)", "Annual Temperature Trend (dashed = linear trend)", "01_temperature_trend.png")
    _trend_plot(annual, "rain_total", "Total rainfall (mm)", "Annual Rainfall Trend (dashed = linear trend)", "02_rainfall_trend.png")

    # Seasonal pattern: monthly temperature + monthly rainfall
    mt = df.groupby(["city", "month"])["temperature_c"].mean().unstack(0)
    mr = df.groupby(["city", "year", "month"])["rainfall_mm"].sum().groupby(["city", "month"]).mean().unstack(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for city in mt.columns:
        a1.plot(mt.index, mt[city], "o-", color=CITY_COLORS.get(city), label=city)
    a1.set(title="Monthly Mean Temperature", xlabel="Month", ylabel="°C", xticks=range(1, 13))
    a1.legend()
    w = 0.8 / len(mr.columns)
    for i, city in enumerate(mr.columns):
        a2.bar(mr.index + (i - 1) * w, mr[city], w, color=CITY_COLORS.get(city), label=city)
    a2.set(title="Average Monthly Rainfall", xlabel="Month", ylabel="mm", xticks=range(1, 13))
    a2.legend()
    _save(fig, "03_seasonal_pattern.png")

    # Anomaly chart (first city)
    city = df.city.iloc[0]
    g = df[df.city == city]
    f = g[g.flag_temp | g.flag_heatwave]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(g.date, g.temp_z, color=T["ocean"], lw=.5, alpha=.7, label="Temperature z-score")
    ax.scatter(f.date, f.temp_z, color=T["heat"], s=18, zorder=3, label="Flagged anomaly / heatwave")
    for y in (3, -3):
        ax.axhline(y, ls="--", color=T["sand"], lw=1)
    ax.set(title=f"Anomaly Detection – {city}", xlabel="Date", ylabel="z-score")
    ax.legend()
    _save(fig, "04_anomaly_detection.png")

    # Year x month temperature-anomaly heatmap
    piv = g.pivot_table(index="year", columns="month", values="temp_anom")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlBu_r", vmin=-2, vmax=2)
    ax.set(title=f"Temperature Anomaly vs 1995–2009 Normal – {city}", xlabel="Month", ylabel="Year",
           xticks=range(12), xticklabels=range(1, 13), yticks=range(0, len(piv), 3), yticklabels=piv.index[::3])
    ax.grid(False)
    fig.colorbar(im, label="°C")
    _save(fig, "05_temp_anomaly_heatmap.png")

    # Forecast
    cities = list(df.city.unique())
    fig, axes = plt.subplots(len(cities), 1, figsize=(11, 3.6 * len(cities)))
    for ax, c in zip(np.atleast_1d(axes), cities):
        hist = df[df.city == c].set_index("date")["temperature_c"].resample("MS").mean()[-96:]
        p = fc[fc.city == c]
        d = pd.to_datetime(p.date)
        ax.plot(hist.index, hist.values, color=T["ocean"], label="Observed")
        ax.plot(d, p.forecast_c, color=CITY_COLORS.get(c), label="Forecast")
        ax.fill_between(d, p.lower_c, p.upper_c, color=CITY_COLORS.get(c), alpha=.2, label="95% interval")
        ax.set(title=f"{c} – 5-Year Temperature Forecast", ylabel="°C")
        ax.legend(loc="upper left", ncol=3)
    _save(fig, "06_forecast.png")
