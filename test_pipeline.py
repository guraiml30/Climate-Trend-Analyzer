from src.anomaly import detect
from src.data_generator import generate
from src.features import add_features
from src.preprocess import clean
from src.trend import annual_summary, trend_stats


def _prep():
    df, _ = clean(generate())
    return add_features(df)


def test_cleaning_removes_gaps_and_errors():
    df, qc = clean(generate())
    assert df[["temperature_c", "humidity_pct"]].isna().sum().sum() == 0
    assert df.temperature_c.max() < 55 and qc["invalid_values_removed"] > 0


def test_warming_trend_is_positive_and_significant():
    t = trend_stats(annual_summary(_prep()))
    row = t[(t.city == "Delhi") & (t.variable == "temperature_c")].iloc[0]
    assert row.slope_per_decade > 0 and row.significant


def test_detector_finds_injected_events():
    _, anoms, m = detect(_prep())
    assert m["recall_injected"] > 0.6 and len(anoms) > 0
