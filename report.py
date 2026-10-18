"""Writes a plain-language insights report (Markdown)."""
from .config import REPORT_DIR


def write_report(summary, tstats, anoms, fcm):
    L = ["# Climate Trend Analyzer – Insights Report", "",
         f"*Generated {summary['generated_at']} · {summary['period'][0]} → {summary['period'][1]} · {summary['records']:,} daily records (simulated)*",
         "", "## Key findings", ""]
    for city in summary["cities"]:
        t = tstats[(tstats.city == city) & (tstats.variable == "temperature_c")].iloc[0]
        r = tstats[(tstats.city == city) & (tstats.variable == "rainfall_mm")].iloc[0]
        s = lambda x: "significant" if x.significant else "not significant"
        L.append(f"- **{city}**: temperature {t.slope_per_decade:+.2f} °C/decade ({s(t)}, p={t.p_value:.3g}); "
                 f"rainfall {r.slope_per_decade:+.0f} mm/decade ({s(r)}, p={r.p_value:.3g}); "
                 f"{int((anoms.city == city).sum())} anomaly days flagged.")
    m = summary["detection"]
    L += ["", "## Anomaly detection quality",
          f"- Recall on injected extreme events: **{m['recall_injected']:.0%}**",
          f"- Precision vs injected events: {m['precision_vs_injected']:.0%} (lower is expected: natural extremes are also flagged)",
          f"- Agreement with IsolationForest: {m['iforest_overlap']:.0%}",
          "", "## Forecast accuracy (24-month holdout)"]
    L += [f"- {r.city}: MAE {r.holdout_mae_c} °C ({r.model})" for r in fcm.itertuples()]
    L += ["", "## Recommendations",
          "- Use heatwave flags as an early-warning trigger for health and power-demand planning.",
          "- Prioritise cities with significant warming trends for heat-resilience investment.",
          "- Track rainfall variability: unchanged totals can still hide more extreme single-day events."]
    (REPORT_DIR / "insights_report.md").write_text("\n".join(L), encoding="utf-8")
