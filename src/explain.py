"""Interpretability: SHAP attributions and a feature-group ablation.

Two complementary questions:

* *What is the model using?* -- SHAP values give an exact, additive
  decomposition of every individual prediction for a tree ensemble, so they
  answer both "which features matter overall" and "why this flight".
* *What is each group of features actually worth?* -- SHAP measures how much
  the fitted model leans on a feature, which is not the same as how much
  predictive value that feature carries. A feature can look important yet be
  fully redundant with another. The ablation retrains the model with each
  group removed and measures the drop in held-out PR-AUC, which is the number
  that matters for deciding what data is worth collecting.

    python -m src.explain --step shap
    python -m src.explain --step ablation --budget 35   # resumable
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Dict, List

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

from src import models as M
from src.config import FIGURES, METRICS, MODE_A, MODELS, N_JOBS, SEED
from src.pipeline import load_splits, xy

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
ABLATION_FILE = METRICS / "ablation.json"
SHAP_FILE = METRICS / "shap_importance.json"

# Human-readable names for the report and the app.
PRETTY = {
    "te_route": "route historical late rate",
    "te_carrier_dest": "carrier x destination historical rate",
    "te_origin_sched_dep_hour": "origin x hour historical rate",
    "te_dest_sched_dep_hour": "destination x hour historical rate",
    "te_carrier_sched_dep_hour": "carrier x hour historical rate",
    "te_tailnum": "airframe historical late rate",
    "te_carrier": "carrier historical late rate",
    "te_dest": "destination historical late rate",
    "sched_dep_hour": "scheduled departure hour",
    "dep_hour_sin": "departure hour (cyclical, sin)",
    "dep_hour_cos": "departure hour (cyclical, cos)",
    "rotation_slack_min": "schedule slack in the rotation (min)",
    "rotation_gap_min": "gap since previous NYC leg (min)",
    "is_tight_turn": "tight rotation flag",
    "leg_of_day": "leg number of the day",
    "tail_legs_today": "legs the airframe flies today",
    "block_slack_min": "block time vs route median (min)",
    "sched_block_min": "scheduled block time (min)",
    "implied_speed_mph": "implied schedule speed (mph)",
    "origin_hour_deps": "departures from origin this hour",
    "origin_slot15_deps": "departures from origin, 15-min slot",
    "origin_day_deps": "departures from origin today",
    "dest_hour_arrivals": "NYC arrivals into destination this hour",
    "carrier_origin_hour_deps": "carrier departures this hour",
    "carrier_share_of_slot": "carrier share of the hourly slot",
    "nyc_hour_deps": "all NYC departures this hour",
    "wx_precip": "precipitation (in)",
    "wx_precip_3h": "precipitation, previous 3 h",
    "wx_precip_6h": "precipitation, previous 6 h",
    "wx_visib": "visibility (mi)",
    "wx_visib_min_3h": "minimum visibility, previous 3 h",
    "wx_wind_speed": "wind speed (mph)",
    "wx_wind_gust": "wind gust (mph)",
    "wx_wind_gust_max_3h": "max wind gust, previous 3 h",
    "wx_wind_dir": "wind direction (deg)",
    "wx_temp": "temperature (F)",
    "wx_dewp": "dew point (F)",
    "wx_humid": "relative humidity (%)",
    "wx_pressure": "pressure (mb)",
    "wx_pressure_change_3h": "pressure change, previous 3 h",
    "wx_low_visibility": "low-visibility flag",
    "wx_freezing": "freezing flag",
    "wx_is_precipitating": "precipitation flag",
    "plane_age": "aircraft age (years)",
    "day_of_week": "day of week",
    "day_of_year": "day of year",
    "is_holiday_period": "holiday travel period",
    "distance": "distance (mi)",
    "dest_alt": "destination elevation (ft)",
    "sched_arr_hour": "scheduled arrival hour",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Feature groups (used by both the aggregated SHAP view and the ablation)
# ---------------------------------------------------------------------------


def feature_groups(cols: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "weather": [c for c in cols if c.startswith("wx_")],
        "historical rates": [c for c in cols if c.startswith("te_")],
        "congestion": [c for c in cols if c in {
            "origin_hour_deps", "origin_slot15_deps", "origin_day_deps",
            "dest_hour_arrivals", "carrier_origin_hour_deps",
            "carrier_share_of_slot", "nyc_hour_deps"}],
        "rotation": [c for c in cols if c in {
            "rotation_gap_min", "rotation_slack_min", "is_tight_turn",
            "leg_of_day", "tail_legs_today"}],
        "calendar": [c for c in cols if c in {
            "month", "day_of_week", "day_of_year", "week_of_year", "is_weekend",
            "is_holiday_period", "sched_dep_hour", "sched_dep_minute",
            "sched_arr_hour", "is_redeye", "dep_hour_sin", "dep_hour_cos"}],
        "aircraft": [c for c in cols if c in {
            "plane_age", "engines", "seats", "engine", "manufacturer",
            "aircraft_type"}],
        "route & schedule": [c for c in cols if c in {
            "distance", "sched_block_min", "block_slack_min",
            "implied_speed_mph", "dest_lat", "dest_lon", "dest_alt",
            "route_annual_flights", "dest_annual_flights", "carrier", "origin",
            "dest"}],
    }
    assigned = {c for v in groups.values() for c in v}
    leftover = [c for c in cols if c not in assigned]
    if leftover:
        groups["other"] = leftover
    return groups


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def step_shap(sample_size: int = 20000) -> None:
    train, valid, test, manifest = load_splits()
    cols = manifest["features"][MODE_A]
    model = joblib.load(MODELS / "lightgbm.joblib")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(test), size=min(sample_size, len(test)), replace=False)
    sample = test.iloc[idx].reset_index(drop=True)
    X, y = xy(sample, cols)

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # older SHAP returns one array per class
        sv = sv[1]
    log.info("SHAP values for %d flights in %.0fs", len(X), time.time() - t0)

    Xp = X.copy()
    Xp.columns = [pretty(c) for c in X.columns]

    # --- global importance -------------------------------------------
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    order = mean_abs.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(11, 10))
    top = order.head(22)[::-1]
    ax.barh([pretty(i) for i in top.index], top.values,
            color=sns.color_palette("mako", len(top)))
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title("What the model relies on\nTop 22 features, test-period sample",
                 fontsize=15)
    fig.savefig(FIGURES / "16_shap_importance.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- beeswarm ------------------------------------------------------
    plt.figure(figsize=(11, 9))
    shap.summary_plot(sv, Xp, max_display=18, show=False, plot_size=None)
    plt.title("Direction and magnitude of each feature's effect", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES / "17_shap_beeswarm.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()

    # --- grouped contribution -----------------------------------------
    groups = feature_groups(cols)
    grouped = {g: float(mean_abs[[c for c in v if c in mean_abs.index]].sum())
               for g, v in groups.items() if v}
    gser = pd.Series(grouped).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(gser.index, gser.values, color=sns.color_palette("mako", len(gser)))
    total = gser.sum()
    for i, v in enumerate(gser.values):
        ax.text(v, i, f"  {v / total:.0%}", va="center", fontsize=12)
    ax.set_xlabel("summed mean |SHAP| (log-odds)")
    ax.set_title("Attribution by feature family", fontsize=15)
    fig.savefig(FIGURES / "18_shap_by_group.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- dependence plots ---------------------------------------------
    top6 = [c for c in order.index[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for ax, feat in zip(axes.ravel(), top6):
        j = list(X.columns).index(feat)
        xs = X[feat]
        if str(xs.dtype) == "category":
            tmp = pd.DataFrame({"level": xs.astype(str), "shap": sv[:, j]})
            m = tmp.groupby("level")["shap"].mean().sort_values()
            ax.barh(m.index[-12:], m.values[-12:],
                    color=sns.color_palette("mako", min(12, len(m))))
        else:
            ax.scatter(xs, sv[:, j], s=5, alpha=.15, color="#3b6978")
            ax.axhline(0, color="grey", lw=1)
            lo, hi = np.nanpercentile(xs.astype(float), [0.5, 99.5])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                ax.set_xlim(lo, hi)
        ax.set_title(pretty(feat), fontsize=13)
        ax.set_ylabel("SHAP (log-odds)")
    fig.suptitle("How the effect varies with the feature value", y=1.0, fontsize=17)
    fig.tight_layout()
    fig.savefig(FIGURES / "19_shap_dependence.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- individual explanations --------------------------------------
    p = model.predict_proba(X)[:, 1]
    picks = {"highest risk": int(np.argmax(p)), "lowest risk": int(np.argmin(p))}
    median_idx = int(np.argsort(np.abs(p - np.median(p)))[0])
    picks["typical"] = median_idx

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    for ax, (label, i) in zip(axes, picks.items()):
        contrib = pd.Series(sv[i], index=X.columns)
        top_c = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])[::-1]
        colors = ["#c44e52" if v > 0 else "#3b6978" for v in top_c.values]
        ax.barh([pretty(k) for k in top_c.index], top_c.values, color=colors)
        ax.axvline(0, color="black", lw=1)
        row = sample.iloc[i]
        ax.set_title(f"{label}: {row['carrier']}{int(row['flight'])} "
                     f"{row['origin']}->{row['dest']}\n"
                     f"{row['flight_date'].date()} {int(row['sched_dep_hour']):02d}:"
                     f"{int(row['sched_dep_minute']):02d}  |  "
                     f"p = {p[i]:.2f}, actual {'LATE' if y[i] else 'on time'}",
                     fontsize=12)
        ax.tick_params(labelsize=10)
    fig.suptitle("Individual flight explanations (red pushes towards late)",
                 y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES / "20_shap_individual.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- level shifters vs discriminators ------------------------------
    stats = _fig_level_vs_rank(sv, X, train)

    SHAP_FILE.write_text(json.dumps({
        "n_sample": int(len(X)),
        "mean_abs_shap": {k: float(v) for k, v in order.items()},
        "group_share": {k: float(v / total) for k, v in
                        sorted(grouped.items(), key=lambda kv: -kv[1])},
        "out_of_range_features": stats,
    }, indent=2))
    log.info("wrote SHAP figures 16-20, 22")


def _fig_level_vs_rank(sv: np.ndarray, X: pd.DataFrame,
                       train: pd.DataFrame) -> Dict:
    """Separate features that shift the level from features that rank flights.

    Mean |SHAP| -- the standard importance bar chart -- sums two very different
    behaviours. A feature whose test values sit entirely outside the training
    range gets pushed into the model's last bin for *every* test flight and
    contributes a large but nearly constant offset: it moves calibration and
    does nothing for discrimination. A feature that varies flight to flight is
    what actually produces ranking. Plotting the mean against the standard
    deviation of the SHAP values separates the two.
    """
    mean_signed = sv.mean(axis=0)
    sd = sv.std(axis=0)
    names = list(X.columns)

    # Which numeric features are extrapolating beyond what training ever saw?
    out_of_range = {}
    for c in names:
        if c not in train.columns or str(X[c].dtype) == "category":
            continue
        tr_lo, tr_hi = train[c].min(), train[c].max()
        xs = pd.to_numeric(X[c], errors="coerce")
        share = float(((xs < tr_lo) | (xs > tr_hi)).mean())
        if share > 0.5:
            out_of_range[c] = {
                "share_of_test_outside_training_range": share,
                "train_range": [float(tr_lo), float(tr_hi)],
                "test_range": [float(xs.min()), float(xs.max())],
                "mean_shap": float(mean_signed[names.index(c)]),
                "sd_shap": float(sd[names.index(c)]),
            }

    fig, ax = plt.subplots(figsize=(12, 8.5))
    flagged = np.array([c in out_of_range for c in names])
    ax.scatter(sd[~flagged], np.abs(mean_signed)[~flagged], s=70,
               color="#3b6978", label="within training range")
    ax.scatter(sd[flagged], np.abs(mean_signed)[flagged], s=120,
               color="#c44e52", label="test values outside training range")
    lim = max(sd.max(), np.abs(mean_signed).max()) * 1.08
    ax.plot([0, lim], [0, lim], ls=":", color="grey")

    interesting = np.argsort(-(np.abs(mean_signed) + sd))[:12]
    for i in interesting:
        ax.annotate(pretty(names[i]), (sd[i], abs(mean_signed[i])),
                    textcoords="offset points", xytext=(7, 5), fontsize=10)
    ax.set_xlabel("SD of SHAP  ->  separates flights (drives AUC)", fontsize=13)
    ax.set_ylabel("|mean SHAP|  ->  shifts the level (drives calibration)", fontsize=13)
    ax.set_title("Not all 'important' features are useful\n"
                 "Points far above the diagonal move the level without ranking anything",
                 fontsize=14)
    ax.legend(fontsize=11)
    fig.savefig(FIGURES / "22_level_vs_ranking.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_of_range


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def step_ablation(budget_s: float | None) -> bool:
    """Retrain with each feature family removed; report the PR-AUC drop."""
    deadline = None if budget_s is None else time.time() + budget_s
    train, valid, test, manifest = load_splits()
    cols = manifest["features"][MODE_A]
    params = joblib.load(MODELS / "training_context.joblib")["best_params"]
    groups = feature_groups(cols)

    done = {}
    if ABLATION_FILE.exists():
        done = json.loads(ABLATION_FILE.read_text())

    # A targeted probe on top of the family ablations. `day_of_year`,
    # `week_of_year` and `month` are monotone indices of calendar time. Under a
    # temporal split their test values (day 305-365) lie entirely outside the
    # training range (day 1-243), so every test flight lands in the model's
    # last bin and receives the same constant nudge. SHAP ranks day_of_year as
    # the single most influential feature, which makes this worth measuring
    # rather than assuming.
    extra = [
        ("time index (day/week/month)", ["day_of_year", "week_of_year", "month"]),
    ]
    jobs = [("full model", [])] + [(g, v) for g, v in groups.items() if v] + extra
    for label, drop in jobs:
        if label in done:
            continue
        if deadline is not None and time.time() > deadline:
            log.info("budget reached; %d/%d ablations done", len(done), len(jobs))
            ABLATION_FILE.write_text(json.dumps(done, indent=2))
            return False
        keep = [c for c in cols if c not in drop]
        Xtr, ytr = xy(train, keep)
        Xva, yva = xy(valid, keep)
        Xte, yte = xy(test, keep)
        model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        p = model.predict_proba(Xte)[:, 1]
        done[label] = {
            "n_features": len(keep),
            "dropped": drop,
            "test_pr_auc": float(average_precision_score(yte, p)),
            "test_roc_auc": float(roc_auc_score(yte, p)),
            "trees": int(model.best_iteration_),
        }
        ABLATION_FILE.write_text(json.dumps(done, indent=2))
        log.info("%-20s features=%3d  test PR-AUC %.4f", label,
                 len(keep), done[label]["test_pr_auc"])

    _fig_ablation(done)
    return True


def _fig_ablation(done: Dict) -> None:
    full = done["full model"]["test_pr_auc"]
    rows = [(k, v["test_pr_auc"], full - v["test_pr_auc"])
            for k, v in done.items() if k != "full model"]
    rows.sort(key=lambda r: r[2])
    labels = [r[0] for r in rows]
    drops = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#c44e52" if d > 0 else "#3b6978" for d in drops]
    ax.barh(labels, drops, color=colors)
    ax.axvline(0, color="black", lw=1)
    for i, (lab, ap, d) in enumerate(rows):
        ax.text(d + (0.0006 if d >= 0 else -0.0006), i,
                f"{ap:.4f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=11)
    ax.set_xlabel("drop in test PR-AUC when the family is removed")
    ax.set_title(f"Feature-family ablation\nfull model test PR-AUC = {full:.4f}; "
                 "bar labels are the ablated model's PR-AUC", fontsize=14)
    fig.savefig(FIGURES / "21_ablation.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    log.info("wrote 21_ablation.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="shap", choices=["shap", "ablation"])
    ap.add_argument("--budget", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    if args.step == "shap":
        step_shap()
    else:
        sys.exit(0 if step_ablation(args.budget) else 2)
