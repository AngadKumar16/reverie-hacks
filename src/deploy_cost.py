"""What does it actually cost to run this every day?

Section 11.1 of the report gives a table of operating requirements -- model
size, retrain time, "a cron job, not a service". Those were estimates written
from the outside. This module measures them, on the machine the report was
produced on, and writes the numbers it finds.

The point is not that the numbers are impressive. It is that "this is cheap to
operate" is a claim about deployment that a datathon project can actually
verify, unlike "this would save an airline $3M", which it cannot. So we verify
the part that is verifiable and label the rest.

What is measured
----------------
* **Artefact size** -- every file the deployment needs, on disk.
* **Cold start** -- loading the booster and the feature contract from disk.
* **Feature build** -- turning one day of raw schedule + weather rows into a
  model frame, which is the part people forget when they time "inference".
* **Scoring** -- one day of New York departures, repeated, reported as median
  and p95 rather than a mean, because a mean over a handful of runs hides the
  tail that an on-call engineer actually cares about.
* **Peak memory** -- resident set growth across a scoring pass.
* **Training** -- total core-seconds, read from the search logs the runs wrote.

What is projected, and labelled as such
---------------------------------------
Scaling to more airports is a multiplication, not a measurement: this dataset
has three origins and no way to observe a fourth. So the scale table sweeps a
volume multiplier rather than asserting a US-wide flight count, and the daily
cost is the measured per-row time times that multiplier. Energy is the measured
CPU time times a stated power assumption, swept, in the same spirit as the
impact model in section 9.

    python -m src.deploy_cost

Writes `reports/metrics/deployment.json` and figure 31.
"""
from __future__ import annotations

import json
import logging
import platform
import resource
import sys
import time
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import features as F
from src.config import (DATA_PROCESSED, FIGURES, METRICS, MODE_A, MODELS,
                        N_JOBS)
from src.data_loader import load_tables
from src.pipeline import load_splits, xy

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
ACCENT = "#c44e52"
BLUE = "#3b6978"

RESULTS_FILE = METRICS / "deployment.json"

# Repeats for the latency measurements. Small because each pass scores a whole
# day; enough to get a median and a p95 that are not single-sample noise.
N_REPEATS = 30

# The files a production deployment genuinely needs: the pre-flight classifier
# and the feature contract. Everything else in models/ is analysis.
DEPLOY_ARTEFACTS = ["lightgbm.joblib", "training_context.joblib"]

# Volume multipliers for the scale projection. 1x is the three NYC airports.
SCALE_MULTIPLIERS = [1, 5, 10, 30, 100]

# Assumptions for the energy figure. Both swept; neither is measured here.
CPU_WATTS_PER_CORE = [5.0, 15.0, 30.0]
GRID_KG_CO2_PER_KWH = 0.369  # US average, EPA eGRID 2022.


def _maxrss_mb() -> float:
    """Peak resident set size in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2) if sys.platform == "darwin" else rss / 1024


def _percentiles(times: List[float]) -> Dict[str, float]:
    a = np.array(times, dtype=float)
    return {
        "median_ms": float(np.median(a) * 1000),
        "p95_ms": float(np.quantile(a, 0.95) * 1000),
        "min_ms": float(a.min() * 1000),
        "max_ms": float(a.max() * 1000),
    }


def measure_artefacts() -> Dict[str, object]:
    files = {}
    for p in sorted(MODELS.glob("*.joblib")):
        files[p.name] = round(p.stat().st_size / 1024 ** 2, 3)
    deploy_mb = sum(files.get(n, 0.0) for n in DEPLOY_ARTEFACTS)
    return {
        "all_artefacts_mb": files,
        "all_artefacts_total_mb": round(sum(files.values()), 2),
        "deployable_subset": DEPLOY_ARTEFACTS,
        "deployable_subset_mb": round(deploy_mb, 3),
    }


def measure_cold_start() -> Dict[str, float]:
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        joblib.load(MODELS / "lightgbm.joblib")
        joblib.load(MODELS / "training_context.joblib")
        times.append(time.perf_counter() - t0)
    return _percentiles(times)


def measure_feature_build(tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Time the feature pipeline on one day of flights.

    Rebuilt from the raw tables, because that is what a daily job does: it does
    not get a cached parquet, it gets a timetable and a weather feed.
    """
    flights = tables["flights"]
    day = flights[(flights["year"] == 2013) & (flights["month"] == 12)
                  & (flights["day"] == 22)]
    subset = dict(tables)
    subset["flights"] = day

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        F.build_feature_frame(subset)
        times.append(time.perf_counter() - t0)
    out = _percentiles(times)
    out["n_flights"] = int(len(day))
    out["date"] = "2013-12-22"
    out["ms_per_flight"] = out["median_ms"] / max(len(day), 1)
    return out


def measure_scoring(test: pd.DataFrame, feats: List[str]) -> Dict[str, object]:
    model = joblib.load(MODELS / "lightgbm.joblib")
    day = test[(test["month"] == 12) & (test["day"] == 22)]
    Xday, _ = xy(day, feats)
    Xone, _ = xy(day.head(1), feats)

    model.predict_proba(Xday)          # warm up; the first call pays import costs

    rss_before = _maxrss_mb()
    day_times, one_times = [], []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        model.predict_proba(Xday)
        day_times.append(time.perf_counter() - t0)
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        model.predict_proba(Xone)
        one_times.append(time.perf_counter() - t0)
    rss_after = _maxrss_mb()

    full_t0 = time.perf_counter()
    Xall, _ = xy(test, feats)
    model.predict_proba(Xall)
    full_s = time.perf_counter() - full_t0

    day_stats = _percentiles(day_times)
    return {
        "one_day": {**day_stats, "n_flights": int(len(day)),
                    "date": "2013-12-22",
                    "us_per_flight": 1000 * day_stats["median_ms"] / max(len(day), 1)},
        "single_flight": _percentiles(one_times),
        "whole_test_period": {
            "n_flights": int(len(test)),
            "seconds": round(full_s, 3),
            "flights_per_second": round(len(test) / full_s),
        },
        "peak_rss_mb": round(rss_after, 1),
        "rss_growth_during_scoring_mb": round(rss_after - rss_before, 1),
        "n_jobs": N_JOBS,
    }


def measure_training_cost() -> Dict[str, object]:
    """Total training effort, read from what the search runs recorded.

    Only the LightGBM search logs a per-draw wall time, so the XGBoost cross-
    check and the final fits are not included and the total is a documented
    lower bound rather than a guess dressed as a measurement.
    """
    search = METRICS / "lgbm_search.json"
    if not search.exists():
        return {"available": False}
    draws = json.loads(search.read_text())
    secs = [d.get("seconds", 0.0) for d in draws]
    total = float(sum(secs))
    return {
        "available": True,
        "measured_component": "LightGBM random search (40 draws x 3 CV folds)",
        "n_draws": len(draws),
        "wall_seconds": round(total, 1),
        "wall_minutes": round(total / 60, 2),
        "median_draw_seconds": round(float(np.median(secs)), 1),
        "core_seconds": round(total * N_JOBS, 1),
        "core_hours": round(total * N_JOBS / 3600, 3),
        "excludes": ["XGBoost search", "final fits", "severity/tier/quantile "
                     "heads", "cancellation models"],
        "note": "lower bound: only the step that records per-draw timings",
    }


def project_scale(scoring: Dict, features: Dict) -> Dict[str, object]:
    """Cost at N times the New York volume.

    A projection, not a measurement -- there is no fourth airport in this
    dataset. Reported as a multiplier sweep so no US-wide flight count has to
    be asserted.
    """
    per_flight_ms = (features["ms_per_flight"]
                     + scoring["one_day"]["us_per_flight"] / 1000)
    base_flights = scoring["one_day"]["n_flights"]
    rows = []
    for m in SCALE_MULTIPLIERS:
        n = base_flights * m
        secs = n * per_flight_ms / 1000
        rows.append({
            "multiplier": m,
            "flights_per_day": int(n),
            "daily_seconds": round(secs, 2),
            "daily_minutes": round(secs / 60, 3),
            "fits_in_one_core_hour": bool(secs < 3600),
        })
    return {
        "ms_per_flight_end_to_end": round(per_flight_ms, 4),
        "basis": "measured feature build + measured scoring, one core-equivalent",
        "rows": rows,
    }


def project_energy(training: Dict, scale: Dict) -> Dict[str, object]:
    """Training and daily-scoring energy, under a swept power assumption."""
    if not training.get("available"):
        return {"available": False}
    core_h = training["core_hours"]
    out = {"available": True, "grid_kg_co2_per_kwh": GRID_KG_CO2_PER_KWH,
           "assumption": "CPU power per core, swept; not measured",
           "training": [], "daily_scoring_at_100x": []}
    daily_core_h = next(r["daily_seconds"] for r in scale["rows"]
                        if r["multiplier"] == 100) / 3600
    for w in CPU_WATTS_PER_CORE:
        kwh = core_h * w / 1000
        out["training"].append({
            "watts_per_core": w,
            "kwh": round(kwh, 4),
            "kg_co2": round(kwh * GRID_KG_CO2_PER_KWH, 4),
        })
        dkwh = daily_core_h * w / 1000
        out["daily_scoring_at_100x"].append({
            "watts_per_core": w,
            "kwh_per_day": round(dkwh, 6),
            "kg_co2_per_year": round(dkwh * 365 * GRID_KG_CO2_PER_KWH, 3),
        })
    return out


def make_figure(res: Dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- where the time goes, per flight ------------------------------
    ax = axes[0]
    fb = res["feature_build"]["ms_per_flight"]
    sc = res["scoring"]["one_day"]["us_per_flight"] / 1000
    ax.bar(["feature\nbuild", "model\nscoring"], [fb, sc],
           color=[BLUE, ACCENT], width=.6)
    for i, v in enumerate([fb, sc]):
        ax.text(i, v + fb * .04, f"{v:.3f} ms", ha="center", fontsize=13,
                weight="bold")
    ax.set_ylim(0, fb * 1.25)
    ax.set_ylabel("milliseconds per flight")
    ax.set_title(f"Building the features costs {fb / sc:.0f}x more\n"
                 "than running the model", fontsize=15)

    # --- scale projection ---------------------------------------------
    ax = axes[1]
    rows = res["scale_projection"]["rows"]
    xs = [r["multiplier"] for r in rows]
    ys = [r["daily_minutes"] for r in rows]
    ax.plot(xs, ys, marker="o", lw=2.8, color=ACCENT)
    for x, y, r in zip(xs, ys, rows):
        ax.annotate(f"{r['flights_per_day']:,}/day", (x, y),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=10)
    ax.axhline(60, color="0.35", ls="--", lw=1.6)
    ax.text(1, 72, "one core-hour per day", fontsize=10, color="0.35")
    # Both axes log: the cost is real but four orders of magnitude below the
    # reference line, and a linear axis would draw it as a flat zero.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(min(ys) * 0.4, 400)
    ax.set_xlabel("multiple of New York departure volume")
    ax.set_ylabel("minutes of compute per day (log)")
    ax.set_title("Scoring stays a cron job,\nnot a service", fontsize=15)

    # --- artefact sizes -------------------------------------------------
    ax = axes[2]
    dep = res["artefacts"]["deployable_subset_mb"]
    allm = res["artefacts"]["all_artefacts_total_mb"]
    ax.bar(["what deployment\nneeds", "every artefact\nthe analysis built"],
           [dep, allm], color=[ACCENT, BLUE], width=.6)
    for i, v in enumerate([dep, allm]):
        ax.text(i, v + allm * .02, f"{v:.1f} MB", ha="center", fontsize=13,
                weight="bold")
    ax.set_ylabel("megabytes on disk")
    ax.set_title("The deployable model is small", fontsize=15)

    fig.suptitle("Measured cost of operating the system", y=1.03, fontsize=17)
    fig.tight_layout()
    fig.savefig(FIGURES / "31_deployment_cost.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote 31_deployment_cost.png")


def main() -> int:
    _, _, test, manifest = load_splits()
    feats = manifest["features"][MODE_A]
    tables = load_tables()

    res: Dict[str, object] = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "n_jobs": N_JOBS,
        },
        "artefacts": measure_artefacts(),
        "cold_start": measure_cold_start(),
        "feature_build": measure_feature_build(tables),
    }
    res["scoring"] = measure_scoring(test, feats)
    res["training"] = measure_training_cost()
    res["scale_projection"] = project_scale(res["scoring"], res["feature_build"])
    res["energy_projection"] = project_energy(res["training"],
                                              res["scale_projection"])

    RESULTS_FILE.write_text(json.dumps(res, indent=2))
    make_figure(res)

    s = res["scoring"]["one_day"]
    log.info("one day (%d flights): %.1f ms median, %.1f ms p95",
             s["n_flights"], s["median_ms"], s["p95_ms"])
    log.info("deployable artefacts: %.1f MB; peak RSS %.0f MB",
             res["artefacts"]["deployable_subset_mb"],
             res["scoring"]["peak_rss_mb"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    sys.exit(main())
