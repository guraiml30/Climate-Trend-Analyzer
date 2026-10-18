"""Forecasting: Holt-Winters (trend + seasonality) on monthly mean temperature."""
from importlib import import_module
import numpy as np
import pandas as pd


class _FallbackModel:
    """Additive Holt-Winters fallback for environments without statsmodels."""

    def __init__(self, series, period=12):
        values = np.asarray(series, dtype=float)
        if len(values) < 2 * period:
            raise ValueError(f"At least {2 * period} observations are required")

        initial = values[:period]
        following = values[period:2 * period]
        level = float(initial.mean())
        trend = float((following.mean() - level) / period)
        seasonal = initial - level
        residuals = []
        alpha, beta, gamma = 0.3, 0.1, 0.1

        for t, value in enumerate(values):
            old_level = level
            old_season = seasonal[t % period]
            if t >= period:
                residuals.append(value - (level + trend + old_season))
            level = alpha * (value - old_season) + (1 - alpha) * (level + trend)
            trend = beta * (level - old_level) + (1 - beta) * trend
            seasonal[t % period] = gamma * (value - level) + (1 - gamma) * old_season

        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.period = period
        self.length = len(values)
        self.index = series.index
        self.resid = np.asarray(residuals)

    def forecast(self, steps):
        values = [self.level + h * self.trend + self.seasonal[(self.length + h - 1) % self.period]
                  for h in range(1, steps + 1)]
        index = pd.date_range(start=self.index[-1], periods=steps + 1, freq="MS")[1:]
        return pd.Series(values, index=index)


def _model(series):
    try:
        exponential_smoothing = import_module("statsmodels.tsa.holtwinters").ExponentialSmoothing
    except ModuleNotFoundError:
        return _FallbackModel(series)
    return exponential_smoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()


def forecast(df: pd.DataFrame, months: int = 60):
    fcs, mets = [], []
    for city, g in df.groupby("city"):
        m = g.set_index("date")["temperature_c"].resample("MS").mean()
        hold = _model(m[:-24]).forecast(24)                      # 24-month holdout test
        mae = float(np.mean(np.abs(hold.values - m[-24:].values)))
        full = _model(m)
        f, sd = full.forecast(months), float(np.std(full.resid))
        fcs.append(pd.DataFrame({"city": city, "date": f.index.strftime("%Y-%m-%d"),
                                 "forecast_c": f.values.round(2),
                                 "lower_c": (f.values - 1.96 * sd).round(2),
                                 "upper_c": (f.values + 1.96 * sd).round(2)}))
        mets.append({"city": city, "holdout_mae_c": round(mae, 3), "model": "Holt-Winters (additive)"})
    return pd.concat(fcs, ignore_index=True), pd.DataFrame(mets)
