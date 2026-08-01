"""Disruption as three outcomes, not two.

The main model drops the 9,430 flights (2.8%) that were cancelled or diverted,
because they have no arrival delay and cannot carry an "arrived late" label.
That was listed as a limitation, and it is the worst kind: cancellations are
the *most* disruptive outcome for a passenger, and they are not randomly
distributed — 5.1% of February departures were cancelled against 0.8% in
October. Dropping them makes the dataset look calmer than the year actually
was, exactly in the storm weeks the model is supposed to warn about.

This module keeps all 336,776 flights and fits three models on the identical
pre-flight feature set:

``cancelled``   a flight that never left (no departure time, no arrival time)
``diverted``    left, but did not arrive where it was meant to
``disrupted``   any of: cancelled, diverted, or arrived >15 minutes late

The third is the one an operations desk actually wants: a single number for
"will this booking go wrong". Reporting it alongside the late-only model shows
whether the dropped rows were hiding anything.

    python -m src.cancellations --budget 35     # resumable
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Dict

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import features as F
from src import models as M
from src.config import (
    DATA_PROCESSED, FIGURES, METRICS, MODE_A, MODELS, TEST_MONTHS,
)
from src.data_loader import load_tables

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
ACCENT = "#c44e52"
BLUE = "#3b6978"
GREEN = "#55a868"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

RESULTS_FILE = METRICS / "disruption.json"
FRAME_FILE = DATA_PROCESSED / "all_flights_features.parquet"

TARGETS = ["is_cancelled", "is_diverted", "is_disrupted"]


def build_full_frame() -> pd.DataFrame:
    """All 336,776 flights, including the ones with no arrival delay."""
    if FRAME_FILE.exists():
        return pd.read_parquet(FRAME_FILE)
    tables = load_tables()
    df = F.build_feature_frame(tables, drop_unlabelled=False)
    df.to_parquet(FRAME_FILE, index=False)
    log.info("built full frame: %d rows (vs %d labelled)",
             len(df), int(df["arr_delay"].notna().sum()))
    return df


def fit_target(df: pd.DataFrame, target: str, params: Dict) -> Dict:
    train, valid, test = F.temporal_split(df)
    # Encodings must be refitted against *this* target -- a carrier's
    # historical late rate says little about its cancellation rate.
    train, (valid, test), enc = F.add_target_encodings(
        train, [valid, test], target=target)
    cols = F.feature_columns(train, MODE_A, enc)

    train = train.sort_values("sched_dep_utc", kind="mergesort")
    valid = valid.sort_values("sched_dep_utc", kind="mergesort")
    Xtr, Xva, Xte = (F.as_model_frame(d, cols) for d in (train, valid, test))
    ytr = train[target].to_numpy()
    yva = valid[target].to_numpy()
    yte = test[target].to_numpy()

    model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    p = model.predict_proba(Xte)[:, 1]

    n_k = int(0.10 * len(p))
    top = np.argsort(-p)[:n_k]
    base = float(yte.mean())
    joblib.dump(model, MODELS / f"lightgbm_{target}.joblib")

    return {
        "target": target,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "base_rate_train": float(ytr.mean()),
        "base_rate_test": base,
        "pr_auc": float(average_precision_score(yte, p)),
        "roc_auc": float(roc_auc_score(yte, p)),
        "brier": float(brier_score_loss(yte, p)),
        "pr_auc_over_base": float(average_precision_score(yte, p)) / base,
        "precision_at_10pct": float(yte[top].mean()),
        "lift_at_10pct": float(yte[top].mean()) / base,
        "recall_at_10pct": float(yte[top].sum() / max(yte.sum(), 1)),
        "trees": int(model.best_iteration_),
    }


def main(budget: float | None) -> int:
    deadline = None if budget is None else time.time() + budget
    params = joblib.load(MODELS / "training_context.joblib")["best_params"]
    df = build_full_frame()

    done: Dict[str, Dict] = {}
    if RESULTS_FILE.exists():
        done = json.loads(RESULTS_FILE.read_text())

    if "descriptives" not in done:
        done["descriptives"] = {
            "n_flights_total": int(len(df)),
            "n_cancelled": int(df["is_cancelled"].sum()),
            "n_diverted": int(df["is_diverted"].sum()),
            "cancellation_rate": float(df["is_cancelled"].mean()),
            "diversion_rate": float(df["is_diverted"].mean()),
            "disruption_rate": float(df["is_disrupted"].mean()),
            "cancellation_rate_by_month": {
                MONTHS[int(k) - 1]: round(float(v), 4) for k, v in
                df.groupby("month")["is_cancelled"].mean().items()},
            "cancellation_rate_by_origin": {
                k: round(float(v), 4) for k, v in
                df.groupby("origin")["is_cancelled"].mean().items()},
            "cancellation_rate_precipitating": float(
                df.loc[df["wx_precip"] > 0, "is_cancelled"].mean()),
            "cancellation_rate_dry": float(
                df.loc[df["wx_precip"] == 0, "is_cancelled"].mean()),
        }
        RESULTS_FILE.write_text(json.dumps(done, indent=2))

    for target in TARGETS:
        if target in done:
            continue
        if deadline and time.time() > deadline:
            log.info("budget reached; rerun to continue")
            return 2
        t0 = time.time()
        done[target] = fit_target(df, target, params)
        RESULTS_FILE.write_text(json.dumps(done, indent=2))
        r = done[target]
        log.info("%-14s base %.4f  PR-AUC %.4f  ROC-AUC %.4f  prec@10%% %.3f "
                 "(lift %.1fx)  %.0fs", target, r["base_rate_test"], r["pr_auc"],
                 r["roc_auc"], r["precision_at_10pct"], r["lift_at_10pct"],
                 time.time() - t0)

    make_figure(df, done)
    return 0


def make_figure(df: pd.DataFrame, done: Dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    m = df.groupby("month")[["is_cancelled", "is_delayed", "is_disrupted"]].mean()
    ax.plot(m.index, m["is_disrupted"], marker="o", lw=2.6, color=ACCENT,
            label="any disruption")
    ax.plot(m.index, m["is_delayed"], marker="s", lw=2.4, color=BLUE,
            label="late > 15 min")
    ax.plot(m.index, m["is_cancelled"], marker="^", lw=2.4, color=GREEN,
            label="cancelled")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, rotation=45)
    ax.set_ylabel("share of scheduled flights")
    ax.set_title("Cancellations peak in February,\nnot with the delay curve",
                 fontsize=14)
    ax.legend(fontsize=10)

    ax = axes[1]
    labels = ["cancelled", "diverted", "any disruption"]
    keys = TARGETS
    base = [done[k]["base_rate_test"] for k in keys]
    prec = [done[k]["precision_at_10pct"] for k in keys]
    x = np.arange(3)
    ax.bar(x - 0.2, base, width=0.4, color=BLUE, label="base rate")
    ax.bar(x + 0.2, prec, width=0.4, color=ACCENT, label="precision in riskiest 10%")
    for i, k in enumerate(keys):
        ax.text(i + 0.2, prec[i] + .01, f"{done[k]['lift_at_10pct']:.1f}x",
                ha="center", fontsize=12, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Held-out targeting performance")
    ax.legend(fontsize=10)

    ax = axes[2]
    roc = [done[k]["roc_auc"] for k in keys]
    bars = ax.bar(labels, roc, color=sns.color_palette("mako", 3))
    for b, v, k in zip(bars, roc, keys):
        ax.text(b.get_x() + b.get_width() / 2, v + .01,
                f"{v:.3f}\nbase {done[k]['base_rate_test']:.1%}",
                ha="center", fontsize=11)
    ax.axhline(0.5, ls=":", color="grey")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("test ROC-AUC")
    ax.set_title("Cancellation is the most predictable\noutcome of the three",
                 fontsize=14)
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Recovering the 9,430 dropped flights: disruption as three "
                 "outcomes", y=1.03, fontsize=17)
    fig.tight_layout()
    fig.savefig(FIGURES / "26_disruption_model.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote 26_disruption_model.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    sys.exit(main(args.budget))
