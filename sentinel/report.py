"""Self-contained HTML fleet-health report (single file, no server needed)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

INK = "#1c2333"
MUTE = "#5b6478"
BLUE = "#2563eb"
AMBER = "#d97706"
RED = "#dc2626"
GREEN = "#059669"
BG_CARD = "#ffffff"

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       background: #f4f6fa; color: #1c2333; line-height: 1.55; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 80px; }
.hero { background: linear-gradient(135deg, #10141f 0%, #1e2b4a 100%); color: #fff;
        border-radius: 16px; padding: 40px 44px; margin-bottom: 28px; }
.hero h1 { font-size: 30px; letter-spacing: -0.5px; }
.hero .tag { color: #9db4e8; font-size: 16px; margin-top: 6px; }
.hero .meta { color: #6b7fa8; font-size: 13px; margin-top: 16px; }
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 28px; }
.kpi { background: #fff; border-radius: 12px; padding: 18px 20px; border: 1px solid #e5e9f2; }
.kpi .v { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
.kpi .l { font-size: 12px; color: #5b6478; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; }
.section { background: #fff; border-radius: 12px; border: 1px solid #e5e9f2;
           padding: 28px 32px; margin-bottom: 24px; }
.section h2 { font-size: 19px; margin-bottom: 4px; letter-spacing: -0.3px; }
.section .sub { color: #5b6478; font-size: 14px; margin-bottom: 18px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
table.q { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.q th { text-align: left; color: #5b6478; font-size: 11.5px; text-transform: uppercase;
             letter-spacing: 0.5px; padding: 8px 10px; border-bottom: 2px solid #e5e9f2; }
table.q td { padding: 9px 10px; border-bottom: 1px solid #eef1f7; vertical-align: top; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 11.5px; font-weight: 600; }
.pill.exception { background: #fee2e2; color: #b91c1c; }
.pill.watch { background: #fef3c7; color: #b45309; }
.pill.data { background: #ede9fe; color: #6d28d9; }
.pill.model { background: #dbeafe; color: #1d4ed8; }
.mono { font-variant-numeric: tabular-nums; }
.note { background: #f8fafc; border-left: 3px solid #2563eb; padding: 14px 18px;
        font-size: 13.5px; color: #3d4557; border-radius: 0 8px 8px 0; margin-top: 16px; }
.footer { color: #8a93a8; font-size: 12.5px; margin-top: 24px; text-align: center; }
.score { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0; }
"""


def _fig_html(fig: go.Figure, height: int = 340) -> str:
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=30, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor="#eef1f7")
    fig.update_yaxes(gridcolor="#eef1f7")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _fleet_charts(queue: pd.DataFrame, signals: pd.DataFrame, n_skus: int) -> str:
    n_exc = int((queue["status"] == "exception").sum()) if len(queue) else 0
    n_watch = int((queue["status"] == "watch").sum()) if len(queue) else 0
    donut = go.Figure(
        go.Pie(
            labels=["Healthy", "Watchlist", "Exception"],
            values=[n_skus - n_exc - n_watch, n_watch, n_exc],
            hole=0.62, sort=False,
            marker=dict(colors=[GREEN, AMBER, RED]),
            textinfo="value",
        )
    )
    donut.update_layout(title=dict(text="Fleet status", font=dict(size=14)))

    counts = signals["check"].value_counts()
    bar = go.Figure(
        go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker_color=BLUE)
    )
    bar.update_layout(
        title=dict(text="Signals by check type", font=dict(size=14)),
        yaxis_title="signals",
    )
    return f'<div class="grid2"><div>{_fig_html(donut)}</div><div>{_fig_html(bar)}</div></div>'


def _queue_table(queue: pd.DataFrame, top: int = 15) -> str:
    rows = []
    for _, r in queue.head(top).iterrows():
        rows.append(
            f"<tr>"
            f"<td><b>{r.get('product', r['sku'])}</b>"
            f"<div style='font-size:11px;color:#8a93a8'>{r['sku']} · {r.get('category', '-')}</div></td>"
            f"<td><span class='pill {r['status']}'>{r['status']}</span></td>"
            f"<td><span class='pill {'data' if r['issue_class'] == 'data quality' else 'model'}'>"
            f"{r['issue_class']}</span></td>"
            f"<td class='mono'>{r['checks_fired']}</td>"
            f"<td class='mono'>{pd.Timestamp(r['first_detected']).date()}</td>"
            f"<td class='mono'>{r['wape_baseline']:.2f} &rarr; {r['wape_now']:.2f}</td>"
            f"<td class='mono'>${r['revenue_at_risk_28d']:,.0f}</td>"
            f"<td style='font-size:12.5px;color:#3d4557'>{r['recommended_action']}</td>"
            f"</tr>"
        )
    return (
        "<table class='q'><thead><tr>"
        "<th>Product</th><th>Status</th><th>Class</th><th>Checks fired</th>"
        "<th>First detected</th><th>WAPE base &rarr; now</th>"
        "<th>Rev at risk (28d)</th><th>Recommended action</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _drilldowns(
    queue: pd.DataFrame,
    fcst: pd.DataFrame,
    signals: pd.DataFrame,
    ground_truth: pd.DataFrame | None,
    n: int = 6,
) -> str:
    gt_map = (
        ground_truth.set_index("sku")[["event_type", "event_date"]].to_dict("index")
        if ground_truth is not None and len(ground_truth)
        else {}
    )
    blocks = []
    picks = queue[queue["status"] == "exception"].head(n)
    for _, r in picks.iterrows():
        sku = r["sku"]
        g = fcst[fcst["sku"] == sku].sort_values("date")
        wk = (
            g.set_index("date")[["actual", "forecast"]]
            .resample("W").sum()
            .reset_index()
        )
        tol = max(float(r["wape_baseline"]) * 1.5, float(r["wape_baseline"]) + 0.10)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=wk["date"], y=wk["forecast"] * (1 + tol),
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=wk["date"], y=np.clip(wk["forecast"] * (1 - tol), 0, None),
                fill="tonexty", fillcolor="rgba(37,99,235,0.10)",
                line=dict(width=0), name="tolerance band", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=wk["date"], y=wk["forecast"], name="forecast (frozen model)",
                       line=dict(color=BLUE, width=2, dash="dot"))
        )
        fig.add_trace(
            go.Scatter(x=wk["date"], y=wk["actual"], name="actual",
                       line=dict(color=INK, width=2.2))
        )
        detect = pd.Timestamp(r["first_detected"])
        fig.add_vline(x=detect, line_color=RED, line_width=2)
        fig.add_annotation(x=detect, y=1.06, yref="paper", text="detected",
                           showarrow=False, font=dict(color=RED, size=11))
        caption = ""
        if sku in gt_map:
            ev = gt_map[sku]
            edate = pd.Timestamp(ev["event_date"])
            fig.add_vline(x=edate, line_color=MUTE, line_width=1.5, line_dash="dash")
            fig.add_annotation(x=edate, y=1.06, yref="paper", text="injected event",
                               showarrow=False, font=dict(color=MUTE, size=11))
            delay = (detect - edate).days
            caption = (
                f"Injected <b>{ev['event_type']}</b> on {edate.date()}, "
                f"detected in <b>{delay} days</b>."
            )
        else:
            caption = "No injected event, flagged from live statistics alone."
        explanation = r.get("explanation", "")
        blocks.append(
            f"<div style='margin-bottom:26px'>"
            f"<h3 style='font-size:15px;margin-bottom:2px'>{r.get('product', sku)} "
            f"<span style='font-weight:400;color:#8a93a8;font-size:12px'>{sku}</span> "
            f"<span class='pill {'data' if r['issue_class'] == 'data quality' else 'model'}'>"
            f"{r['primary_check']}</span></h3>"
            f"<div style='font-size:13px;color:#3d4557;margin-bottom:2px'>{explanation}</div>"
            f"<div style='font-size:12.5px;color:#5b6478'>{caption} "
            f"{r['recommended_action']}.</div>"
            f"{_fig_html(fig, height=300)}</div>"
        )
    return "".join(blocks)


def _scorecard(sc: dict) -> str:
    by_type = pd.DataFrame(sc["by_type"])
    rows = "".join(
        f"<tr><td>{r['event_type']}</td><td class='mono'>{r['events']}</td>"
        f"<td class='mono'>{r['detected']}</td><td class='mono'>{r['recall']:.0%}</td></tr>"
        for _, r in by_type.iterrows()
    )
    delay = sc["median_detection_delay_days"]
    return f"""
    <div class="score">
      <div class="kpi"><div class="v">{sc['recall']:.0%}</div><div class="l">Recall, events caught</div></div>
      <div class="kpi"><div class="v">{sc['precision']:.0%}</div><div class="l">Precision, alarms that were real</div></div>
      <div class="kpi"><div class="v">{delay:.0f}d</div><div class="l">Median detection delay</div></div>
      <div class="kpi"><div class="v">{sc['false_positives']}</div><div class="l">False alarms / 6 months</div></div>
    </div>
    <table class="q"><thead><tr><th>Injected event type</th><th>Events</th><th>Detected</th><th>Recall</th></tr></thead>
    <tbody>{rows}</tbody></table>
    <div class="note">Every drift event was injected with a known date, so these numbers are measured
    against ground truth, not eyeballed. Data-quality breaks (frozen feeds, outages, unit bugs) are
    flagged at the first weekly checkpoint; slow demand drift is confirmed over ~3 checkpoints by design,
    trading a two-week delay for an 80% precision alarm stream that operations teams will actually trust.</div>
    """


def build_report(
    queue: pd.DataFrame,
    signals: pd.DataFrame,
    fcst: pd.DataFrame,
    scorecard: dict | None,
    ground_truth: pd.DataFrame | None,
    n_skus: int,
    out_path: str,
) -> None:
    n_exc = int((queue["status"] == "exception").sum()) if len(queue) else 0
    n_watch = int((queue["status"] == "watch").sum()) if len(queue) else 0
    rev_risk = float(queue.loc[queue["status"] == "exception", "revenue_at_risk_28d"].sum()) if len(queue) else 0
    n_data = int((queue["issue_class"] == "data quality").sum()) if len(queue) else 0
    today = pd.Timestamp(fcst["date"].max()).date()

    score_html = (
        f"<div class='section'><h2>Detection scorecard, measured against ground truth</h2>"
        f"<div class='sub'>28 drift &amp; data-quality events were injected into the simulation "
        f"with known dates. Here is exactly what the monitor caught, missed, and imagined.</div>"
        f"{_scorecard(scorecard)}</div>"
        if scorecard
        else ""
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SKU Sentinel, Fleet Health Report</title>
<style>{CSS}</style>
<script>{get_plotlyjs()}</script>
</head><body><div class="wrap">

<div class="hero">
  <h1>SKU Sentinel</h1>
  <div class="tag">Statistical governance for demand forecasts: tolerance bands, drift tests
  and an impact-ranked exception queue, so {n_skus:,} SKUs don't need {n_skus:,} sign-offs.</div>
  <div class="meta">Fleet health report &nbsp;·&nbsp; monitoring window through {today}
  &nbsp;·&nbsp; weekly checkpoints, 28-day trailing windows &nbsp;·&nbsp; thresholds calibrated
  per SKU on a clean out-of-sample period</div>
</div>

<div class="kpis">
  <div class="kpi"><div class="v">{n_skus:,}</div><div class="l">SKUs monitored</div></div>
  <div class="kpi"><div class="v" style="color:{GREEN}">{n_skus - n_exc - n_watch:,}</div><div class="l">Healthy</div></div>
  <div class="kpi"><div class="v" style="color:{AMBER}">{n_watch}</div><div class="l">Watchlist</div></div>
  <div class="kpi"><div class="v" style="color:{RED}">{n_exc}</div><div class="l">Exceptions</div></div>
  <div class="kpi"><div class="v">${rev_risk:,.0f}</div><div class="l">Revenue at risk (28d)</div></div>
</div>

<div class="section">
  <h2>Fleet overview</h2>
  <div class="sub">{n_exc} exceptions ({n_data} of them data-quality breaks that should be fixed
  in the pipeline, not the model) and {n_watch} SKUs on the watchlist.</div>
  {_fleet_charts(queue, signals, n_skus)}
</div>

<div class="section">
  <h2>Exception queue, ranked by revenue at risk</h2>
  <div class="sub">Governance intensity scales with impact: a human reviews these
  {min(15, n_exc)} rows, not {n_skus:,}. Each row carries a diagnosis and a next action.</div>
  {_queue_table(queue)}
</div>

<div class="section">
  <h2>Exception drill-downs</h2>
  <div class="sub">Weekly actuals vs. the frozen model's forecast. Grey dashed line = where the
  problem actually started (hidden from the monitor); red line = where Sentinel called it.</div>
  {_drilldowns(queue, fcst, signals, ground_truth)}
</div>

{score_html}

<div class="section">
  <h2>How it works</h2>
  <div class="sub" style="margin-bottom:8px">Eight checks, three severity tiers, one queue.</div>
  <div style="font-size:14px;color:#3d4557">
  A frozen baseline forecaster is monitored at weekly checkpoints over trailing 28-day windows.
  Per-SKU tolerance bands are calibrated on a clean out-of-sample period. <b>Data-quality checks</b>
  (frozen values, zero-runs, order-of-magnitude scale breaks) alarm immediately, because acting on a
  forecast fed by a broken pipeline is worse than no forecast. <b>Model-health checks</b> (rolling
  WAPE vs. tolerance, calibrated bias deltas, PSI with a finite-sample null correction, KS tests,
  and a standardized Page-Hinkley sequential drift test) must persist across three consecutive
  checkpoints before alarming, killing alert fatigue at the cost of a two-week confirmation delay.
  Exceptions are ranked by revenue at risk, so review effort follows impact.</div>
</div>

<div class="footer">SKU Sentinel · built by Vinith Kumar · Python, scikit-learn, SciPy, Plotly ·
concept: statistical governance for high-volume forecasting decisions</div>
</div></body></html>"""
    with open(out_path, "w") as f:
        f.write(html)
