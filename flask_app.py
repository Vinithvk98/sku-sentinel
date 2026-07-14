"""SKU Sentinel, Flask web console.

Run `python run_demo.py` first (creates ./output), then:

    python flask_app.py            # http://127.0.0.1:5000

FLASK CONCEPTS USED HERE (learn these, they're the whole framework):

  @app.route(...)       binds a URL to a Python function ("view function")
  render_template(...)  fills a Jinja2 HTML template in ./templates with data
  <sku> in a route      a URL variable, passed as an argument to the function
  jsonify(...)          returns JSON, this is how you build an API
  request               the incoming HTTP request: form fields, uploaded files
  redirect/url_for      send the browser to another route by function name
"""
from __future__ import annotations

import io
import json
import pathlib

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from sentinel.explain import drivers_for_sku, uplift_fleet, uplift_for_sku
from sentinel.models import IMPORT_ERRORS, REGISTRY, run_bench
from sentinel.monitor import run_monitor
from sentinel.simulate import CALIB_END, TRAIN_END
from sentinel.triage import build_queue

BASE = pathlib.Path(__file__).parent
OUT = BASE / "output"

app = Flask(__name__)

# --------------------------------------------------------------- data layer
# Loaded once at startup and kept in memory (fine for a local, single-user
# tool). A production app would use a database + caching instead.
_DEMO: dict = {}
_UPLOAD: dict = {}  # results of the user's last CSV upload


def demo() -> dict:
    if not _DEMO and (OUT / "queue.csv").exists():
        _DEMO["queue"] = pd.read_csv(OUT / "queue.csv", parse_dates=["first_detected"])
        _DEMO["fcst"] = pd.read_csv(OUT / "forecasts.csv", parse_dates=["date"])
        _DEMO["signals"] = pd.read_csv(OUT / "signals.csv", parse_dates=["date"])
        _DEMO["metrics"] = pd.read_csv(OUT / "metrics.csv", parse_dates=["date"])
        _DEMO["gt"] = pd.read_csv(OUT / "ground_truth.csv", parse_dates=["event_date"])
        _DEMO["scorecard"] = json.loads((OUT / "scorecard.json").read_text())
        _DEMO["panel"] = pd.read_csv(OUT / "panel.csv", parse_dates=["date"])
        cat_path = OUT / "catalog.csv"
        _DEMO["labels"] = (
            pd.read_csv(cat_path).set_index("sku")["name"].to_dict()
            if cat_path.exists() else {}
        )
    return _DEMO


_BENCH_CACHE: dict = {}  # (sku, model) -> result, so re-runs are instant


def weekly_series(fcst: pd.DataFrame, sku: str) -> dict:
    """Weekly actual/forecast arrays for one SKU, the payload Plotly.js plots."""
    g = fcst[fcst["sku"] == sku].sort_values("date")
    wk = g.set_index("date")[["actual", "forecast"]].resample("W").sum().reset_index()
    return {
        "dates": wk["date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": wk["actual"].round(1).tolist(),
        "forecast": wk["forecast"].round(1).tolist(),
    }


def kpis(queue: pd.DataFrame, n_skus: int) -> dict:
    n_exc = int((queue["status"] == "exception").sum()) if len(queue) else 0
    n_watch = int((queue["status"] == "watch").sum()) if len(queue) else 0
    rev = float(queue.loc[queue["status"] == "exception", "revenue_at_risk_28d"].sum()) if len(queue) else 0.0
    return {"n_skus": n_skus, "healthy": n_skus - n_exc - n_watch,
            "watch": n_watch, "exceptions": n_exc, "rev_at_risk": rev}


# -------------------------------------------------------------------- pages
@app.route("/")
def landing():
    """Marketing-style home page. Works even before run_demo.py has run."""
    d = demo()
    stats = None
    if d:
        sc = d["scorecard"]
        q = d["queue"]
        stats = {
            "recall": sc["recall"], "precision": sc["precision"],
            "false_positives": sc["false_positives"],
            "n_skus": d["fcst"]["sku"].nunique(),
            "rev_at_risk": float(
                q.loc[q["status"] == "exception", "revenue_at_risk_28d"].sum()
            ) if len(q) else 0.0,
        }
    return render_template("landing.html", stats=stats)


@app.route("/queue")
def index():
    d = demo()
    if not d:
        return render_template("index.html", ready=False)
    q = d["queue"]
    return render_template(
        "index.html", ready=True,
        kpis=kpis(q, d["fcst"]["sku"].nunique()),
        rows=q.to_dict(orient="records"),
    )


@app.route("/sku/<sku>")
def sku_page(sku: str):
    src = request.args.get("src", "demo")  # ?src=upload after a CSV upload
    d = _UPLOAD if src == "upload" else demo()
    if not d:
        return redirect(url_for("index"))
    row = d["queue"][d["queue"]["sku"] == sku]
    row = row.to_dict(orient="records")[0] if len(row) else None
    event = None
    if "gt" in d:
        ev = d["gt"][d["gt"]["sku"] == sku]
        if len(ev):
            event = {"date": ev["event_date"].iloc[0].strftime("%Y-%m-%d"),
                     "type": ev["event_type"].iloc[0]}
    m = d.get("metrics")
    m = m[m["sku"] == sku] if m is not None else None
    tol = None
    if m is not None and len(m):
        tol = {
            "dates": m["date"].dt.strftime("%Y-%m-%d").tolist(),
            "wape": m["wape_28"].round(3).tolist(),
            "tolerance": m["wape_tolerance"].round(3).tolist(),
        }
    sigs = d.get("signals")
    sigs = (sigs[sigs["sku"] == sku].to_dict(orient="records")
            if sigs is not None else [])
    return render_template(
        "sku.html", sku=sku, src=src, row=row, event=event,
        series=weekly_series(d["fcst"], sku), tolerance=tol, signals=sigs,
        all_skus=sorted(d["fcst"]["sku"].unique().tolist()),
    )


@app.route("/scorecard")
def scorecard():
    d = demo()
    if not d:
        return redirect(url_for("index"))
    return render_template("scorecard.html", sc=d["scorecard"])


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """GET shows the form; POST processes the uploaded CSV. One route, two verbs."""
    if request.method == "GET":
        return render_template("upload.html", result=None, error=None)

    try:
        f = request.files.get("csv_file")
        if f is None or f.filename == "":
            raise ValueError("No file selected.")
        calib_frac = float(request.form.get("calib_frac", 0.5))

        user = pd.read_csv(io.BytesIO(f.read()))
        user.columns = [c.strip().lower() for c in user.columns]
        missing = {"sku", "date", "actual", "forecast"} - set(user.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        user["date"] = pd.to_datetime(user["date"], errors="coerce")
        user["actual"] = pd.to_numeric(user["actual"], errors="coerce")
        user["forecast"] = pd.to_numeric(user["forecast"], errors="coerce")
        user = user.dropna(subset=["date", "actual", "forecast"]).sort_values(["sku", "date"])

        split = user.groupby("sku")["date"].transform(
            lambda dts: dts.quantile(calib_frac, interpolation="nearest")
        )
        user["window"] = (user["date"] > split).map({False: "calibration", True: "monitoring"})
        days = user.groupby(["sku", "window"])["date"].nunique().unstack(fill_value=0)
        ok = days[(days.get("calibration", 0) >= 56) & (days.get("monitoring", 0) >= 35)].index
        skipped = user["sku"].nunique() - len(ok)
        user = user[user["sku"].isin(ok)]
        if user.empty:
            raise ValueError("Each product needs ~8 weeks of calibration history "
                             "and ~5 weeks to monitor. Not enough daily data found.")

        catalog = pd.DataFrame({"sku": user["sku"].unique(), "price": 1.0})
        signals, metrics = run_monitor(user, catalog)
        queue = build_queue(signals, metrics, user, catalog)

        _UPLOAD.clear()
        _UPLOAD.update({"fcst": user, "queue": queue, "signals": signals, "metrics": metrics})
        return render_template(
            "upload.html", error=None,
            result={
                "kpis": kpis(queue, user["sku"].nunique()),
                "rows": queue.to_dict(orient="records") if len(queue) else [],
                "skipped": skipped,
            },
        )
    except Exception as exc:
        return render_template("upload.html", result=None, error=str(exc))


@app.route("/sample.csv")
def sample_csv():
    d = demo()
    if not d:
        return redirect(url_for("index"))
    skus = d["fcst"]["sku"].unique()[:3]
    sample = d["fcst"][d["fcst"]["sku"].isin(skus)][["sku", "date", "actual", "forecast"]]
    return (sample.to_csv(index=False),
            {"Content-Type": "text/csv",
             "Content-Disposition": "attachment; filename=sentinel_sample.csv"})


@app.route("/explain")
def explain_page():
    d = demo()
    if not d:
        return redirect(url_for("index"))
    skus = sorted(d["panel"]["sku"].unique().tolist())
    return render_template("explain.html", skus=skus, labels=d.get("labels", {}))


@app.route("/api/explain/<sku>")
def api_explain(sku: str):
    d = demo()
    if not d:
        return jsonify({"error": "run run_demo.py first"}), 404
    g = d["panel"][d["panel"]["sku"] == sku]
    if g.empty:
        return jsonify({"error": "unknown sku"}), 404
    return jsonify({"sku": sku, "drivers": drivers_for_sku(g),
                    "uplift": uplift_for_sku(g)})


@app.route("/api/uplift")
def api_uplift():
    d = demo()
    if not d:
        return jsonify({"error": "run run_demo.py first"}), 404
    if "uplift_fleet" not in _DEMO:  # ~1s for 200 SKUs; computed once
        cat = pd.read_csv(OUT / "catalog.csv") if (OUT / "catalog.csv").exists() else None
        _DEMO["uplift_fleet"] = uplift_fleet(d["panel"], cat)
    return jsonify(_DEMO["uplift_fleet"])


@app.route("/models")
def models_page():
    d = demo()
    if not d:
        return redirect(url_for("index"))
    skus = sorted(d["panel"]["sku"].unique().tolist())
    return render_template("models.html", registry=REGISTRY, skus=skus,
                           labels=d.get("labels", {}), import_errors=IMPORT_ERRORS)


@app.route("/api/bench/<sku>")
def api_bench(sku: str):
    """Run the requested models for one SKU and return the leaderboard.

    Called from JavaScript with fetch(), the page never reloads. Results are
    cached per (sku, model), so adding one more model reuses the others.
    """
    d = demo()
    if not d:
        return jsonify({"error": "run run_demo.py first"}), 404
    names = [m for m in request.args.get("models", "").split(",") if m in REGISTRY]
    if not names:
        return jsonify({"error": "no valid models requested"}), 400

    todo = [m for m in names if (sku, m) not in _BENCH_CACHE]
    if todo:
        g = d["panel"][d["panel"]["sku"] == sku]
        fresh = run_bench(g, todo, TRAIN_END, CALIB_END)
        for m in todo:
            _BENCH_CACHE[(sku, m)] = fresh["models"][m]
        _BENCH_CACHE[(sku, "_meta")] = {"dates": fresh["dates"], "actual": fresh["actual"]}

    meta = _BENCH_CACHE[(sku, "_meta")]
    models = {m: _BENCH_CACHE[(sku, m)] for m in names}
    scored = {m: v for m, v in models.items() if v.get("wape") is not None}
    champion = min(scored, key=lambda m: scored[m]["wape"]) if scored else None
    return jsonify({"dates": meta["dates"], "actual": meta["actual"],
                    "models": models, "champion": champion})


# ---------------------------------------------------------------- JSON API
# The same data as the pages, but machine-readable, this is how another
# system (or a JS frontend) would consume Sentinel.
@app.route("/api/queue")
def api_queue():
    d = demo()
    return jsonify(d["queue"].to_dict(orient="records") if d else [])


@app.route("/api/sku/<sku>")
def api_sku(sku: str):
    d = demo()
    if not d:
        return jsonify({"error": "run run_demo.py first"}), 404
    return jsonify(weekly_series(d["fcst"], sku))


@app.route("/api/scorecard")
def api_scorecard():
    d = demo()
    return jsonify(d["scorecard"] if d else {})


if __name__ == "__main__":
    # debug=True -> auto-reload on code changes + error pages in the browser.
    # Never leave debug on in a deployed app.
    app.run(debug=True)
