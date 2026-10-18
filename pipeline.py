"""End-to-end pipeline: simulate -> clean -> features -> trends -> anomalies -> forecast -> charts -> report."""
import json
import time
import warnings
from datetime import datetime

from .anomaly import detect
from .config import END, OUT_DIR, PROC_DIR, RAW_DIR, START
from .data_generator import generate
from .features import add_features
from .forecast import forecast
from .preprocess import clean
from .report import write_report
from .trend import annual_summary, seasonal_summary, trend_stats
from .visualize import plot_all


def run_pipeline(verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    log("[1/7] Simulating climate data ...")
    raw = generate()
    raw.to_csv(RAW_DIR / "climate_raw.csv", index=False)
    log(f"      {len(raw):,} rows, {raw.city.nunique()} cities")

    log("[2/7] Cleaning data ...")
    df, qc = clean(raw)
    log(f"      removed {qc['invalid_values_removed']} sensor errors, filled {qc['missing_values_filled']} gaps")

    log("[3/7] Engineering features ...")
    df = add_features(df)

    log("[4/7] Analysing long-term trends ...")
    annual = annual_summary(df)
    tstats = trend_stats(annual)
    seasonal = seasonal_summary(df)
    if verbose:
        print(tstats.to_string(index=False))

    log("[5/7] Detecting anomalies ...")
    df, anoms, metrics = detect(df)
    log(f"      flagged {metrics['flagged_days']} days | recall on injected events {metrics['recall_injected']:.0%}")

    log("[6/7] Forecasting 5 years ahead ...")
    fc, fcm = forecast(df)

    log("[7/7] Creating charts and report ...")
    df.to_csv(PROC_DIR / "climate_features.csv", index=False)
    annual.to_csv(OUT_DIR / "annual_summary.csv", index=False)
    tstats.to_csv(OUT_DIR / "trend_stats.csv", index=False)
    seasonal.to_csv(OUT_DIR / "seasonal_summary.csv", index=False)
    anoms.to_csv(OUT_DIR / "anomalies.csv", index=False)
    fc.to_csv(OUT_DIR / "forecast.csv", index=False)
    fcm.to_csv(OUT_DIR / "forecast_metrics.csv", index=False)
    plot_all(df, annual, fc)

    warming = tstats[tstats.variable == "temperature_c"].set_index("city")["slope_per_decade"]
    summary = dict(project="Climate Trend Analyzer", period=[START, END], cities=sorted(df.city.unique().tolist()),
                   records=int(len(df)), quality=qc, detection=metrics,
                   warming_per_decade={k: float(v) for k, v in warming.items()},
                   forecast_mae={r.city: float(r.holdout_mae_c) for r in fcm.itertuples()},
                   generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, tstats, anoms, fcm)
    log(f"Done in {time.time() - t0:.1f}s. Outputs -> outputs/, images/, reports/")
    return summary
