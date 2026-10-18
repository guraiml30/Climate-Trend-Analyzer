"""Long-term trend analysis (OLS regression on annual values)."""
import pandas as pd
import math


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""
    max_iterations, epsilon, tiny = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        term = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        term = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1 - factor * _beta_continued_fraction(b, a, 1 - x) / b


def _slope_p_value(t_stat: float, degrees_of_freedom: int) -> float:
    x = degrees_of_freedom / (degrees_of_freedom + t_stat * t_stat)
    return _regularized_beta(x, degrees_of_freedom / 2, 0.5)


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["city", "year"])
              .agg(temp_mean=("temperature_c", "mean"), temp_max=("temperature_c", "max"),
                   rain_total=("rainfall_mm", "sum"), humidity_mean=("humidity_pct", "mean"))
              .reset_index().round(2))


def trend_stats(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in annual.groupby("city"):
        for var, col in (("temperature_c", "temp_mean"), ("rainfall_mm", "rain_total")):
            values = g[["year", col]].dropna()
            x, y = values["year"].astype(float), values[col].astype(float)
            n = len(values)
            x_mean, y_mean = x.mean(), y.mean()
            ss_x = ((x - x_mean) ** 2).sum()
            ss_y = ((y - y_mean) ** 2).sum()
            if n < 2 or ss_x == 0:
                slope, r2, p_value = float("nan"), float("nan"), float("nan")
            else:
                slope = (((x - x_mean) * (y - y_mean)).sum()) / ss_x
                residual_ss = ((y - (y_mean + slope * (x - x_mean))) ** 2).sum()
                r2 = 1 - residual_ss / ss_y if ss_y else 1.0
                if n <= 2:
                    p_value = float("nan")
                elif residual_ss == 0:
                    p_value = 0.0 if slope != 0 else 1.0
                else:
                    standard_error = math.sqrt((residual_ss / (n - 2)) / ss_x)
                    p_value = _slope_p_value(slope / standard_error, n - 2)
            rows.append(dict(city=city, variable=var, slope_per_decade=round(slope * 10, 3),
                             p_value=round(p_value, 5), r2=round(r2, 3),
                             significant=bool(p_value < 0.05)))
    return pd.DataFrame(rows)


def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["city", "season"])
              .agg(avg_temp_c=("temperature_c", "mean"), avg_daily_rain_mm=("rainfall_mm", "mean"))
              .round(2).reset_index())
