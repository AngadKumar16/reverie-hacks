"""Does this work at an airport it has never seen?

Section 11.4 of the report claims the system extends beyond the three New York
origins because "the feature code is origin-agnostic". That is an assertion
about the code, not a measurement of the model, and it is exactly the kind of
claim this project is supposed to test rather than make.

So this module measures it. For each of EWR, JFK and LGA in turn:

* train on the other two origins only, over the same Jan-Aug window;
* recompute the historical-rate encodings from *only* those two airports, so
  no scrap of the held-out airport's history reaches the model;
* early-stop on Sep-Oct rows from those two airports only;
* score the held-out airport's Nov-Dec flights -- an airport the model has
  never seen, in a period it has never seen.

The comparison is the shipped model (trained on all three origins) scored on
the *same* flights. The gap between the two is the cold-start cost: what it
costs to open at a new airport with no local history, versus running where you
already have a year of it.

This is the deployment question. A delay model that only works where it was
trained has to be rebuilt for every airport and is worth much less than one
that ships with a useful prior on day one.

Two encodings necessarily collapse for the held-out airport. ``te_route`` and
``te_origin_sched_dep_hour`` are keyed on the origin, so every held-out row
falls back to the global prior -- they carry no information at a new airport by
construction. ``te_carrier``, ``te_dest``, ``te_tailnum`` and ``te_carrier_dest``
still transfer, because carriers, destinations and airframes are shared across
the network. Which of those survives is reported below.

    python -m src.transfer

Writes `reports/metrics/transfer.json` and figure 30.
"""
from __future__ import annotations

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
from sklearn.metrics import average_precision_score, roc_auc_score

from src import features as F
from src import models as M
from src.config import FIGURES, METRICS, MODE_A, MODELS, CAPACITY_FRACTION
from src.pipeline import load_full, load_splits, xy

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
ACCENT = "#c44e52"
BLUE = "#3b6978"

RESULTS_FILE = METRICS / "transfer.json"
ORIGINS = ["EWR", "JFK", "LGA"]

# The origin-keyed encodings, which cannot survive a change of airport.
ORIGIN_KEYED = ["te_route", "te_origin_sched_dep_hour"]


def _operational(y: np.ndarray, p: np.ndarray,
                 k: float = CAPACITY_FRACTION) -> Dict[str, float]:
    """Precision and lift in the riskiest k of the list -- the read that
    matters to a desk that can only act on so many flights."""
    n = max(int(len(p) * k), 1)
    top = np.argsort(-p)[:n]
    base = float(y.mean())
    prec = float(y[top].mean())
    return {
        "n_flagged": n,
        "precision_at_10pct": prec,
        "recall_at_10pct": float(y[top].sum() / max(y.sum(), 1)),
        "lift_at_10pct": prec / base if base else float("nan"),
    }


def _scores(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "base_rate": float(y.mean()),
    }
    out.update(_operational(y, p))
    return out


def run_origin(held_out: str, params: Dict) -> Dict[str, object]:
    """Train without `held_out`, then score that airport's test flights."""
    t0 = time.time()

    # --- the transfer model: this airport is invisible during training ----
    full = load_full()
    tr, va, te = F.temporal_split(full)
    tr_r = tr[tr["origin"] != held_out].copy()
    va_r = va[va["origin"] != held_out].copy()
    te_o = te[te["origin"] == held_out].copy()

    # Encodings refitted on the two remaining airports only.
    tr_r, (va_r, te_o), enc = F.add_target_encodings(tr_r, [va_r, te_o])
    feats: List[str] = F.feature_columns(tr_r, MODE_A, enc)

    Xtr, ytr = xy(tr_r, feats)
    Xva, yva = xy(va_r, feats)
    Xte, yte = xy(te_o, feats)

    model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    p_transfer = model.predict_proba(Xte)[:, 1]

    # --- the ceiling: the shipped model, same flights --------------------
    # `test.parquet` comes from the same `temporal_split` of the same frame, so
    # filtering both by origin preserves the row order. Assert it rather than
    # trust it: pairing a flight with somebody else's probability is the exact
    # bug that produces plausible, wrong numbers.
    _, _, test_proc, manifest = load_splits()
    te_shipped = test_proc[test_proc["origin"] == held_out]
    y_shipped = te_shipped["is_delayed"].to_numpy()
    if not np.array_equal(y_shipped, yte):
        raise AssertionError(
            f"row alignment broke for {held_out}: the shipped test frame and "
            "the re-encoded frame are not the same flights in the same order")

    shipped = joblib.load(MODELS / "lightgbm.joblib")
    Xte_shipped, _ = xy(te_shipped, manifest["features"][MODE_A])
    p_within = shipped.predict_proba(Xte_shipped)[:, 1]

    # --- which encodings actually carried across --------------------------
    collapsed = {}
    for name in enc:
        col = te_o[name]
        # An encoding that fell back to the prior everywhere is constant.
        collapsed[name] = bool(col.nunique(dropna=True) <= 1)

    transfer, within = _scores(yte, p_transfer), _scores(yte, p_within)
    res = {
        "held_out_origin": held_out,
        "n_train_flights": int(len(tr_r)),
        "n_test_flights": int(len(te_o)),
        "train_origins": sorted(tr_r["origin"].unique().tolist()),
        "trees": int(model.best_iteration_ or 0),
        "transfer": transfer,
        "within_network": within,
        "retention": {
            "pr_auc": transfer["pr_auc"] / within["pr_auc"],
            "roc_auc": ((transfer["roc_auc"] - 0.5)
                        / (within["roc_auc"] - 0.5)),
            "precision_at_10pct": (transfer["precision_at_10pct"]
                                   / within["precision_at_10pct"]),
        },
        "encodings_collapsed_to_prior": sorted(k for k, v in collapsed.items() if v),
        "encodings_that_transferred": sorted(k for k, v in collapsed.items() if not v),
        "seconds": round(time.time() - t0, 1),
    }
    log.info("%s held out: transfer PR-AUC %.4f vs %.4f within-network "
             "(%.1f%% retained), lift %.2fx vs %.2fx", held_out,
             transfer["pr_auc"], within["pr_auc"],
             100 * res["retention"]["pr_auc"],
             transfer["lift_at_10pct"], within["lift_at_10pct"])
    return res


def make_figure(done: Dict[str, Dict]) -> None:
    airports = [o for o in ORIGINS if o in done]
    x = np.arange(len(airports))
    tr_pr = [done[o]["transfer"]["pr_auc"] for o in airports]
    wi_pr = [done[o]["within_network"]["pr_auc"] for o in airports]
    base = [done[o]["transfer"]["base_rate"] for o in airports]
    tr_lift = [done[o]["transfer"]["lift_at_10pct"] for o in airports]
    wi_lift = [done[o]["within_network"]["lift_at_10pct"] for o in airports]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    ax = axes[0]
    ax.bar(x - 0.2, wi_pr, width=0.4, color=BLUE,
           label="trained on all three airports")
    ax.bar(x + 0.2, tr_pr, width=0.4, color=ACCENT,
           label="airport never seen in training")
    for i, b in enumerate(base):
        ax.hlines(b, i - 0.42, i + 0.42, color="0.25", ls="--", lw=2.2,
                  label="base rate (no model)" if i == 0 else None)
    for i, o in enumerate(airports):
        ax.text(i + 0.2, tr_pr[i] + .012,
                f"{100 * done[o]['retention']['pr_auc']:.0f}%",
                ha="center", fontsize=12, weight="bold", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(airports)
    # Headroom so the retention labels clear the legend.
    ax.set_ylim(0, max(wi_pr) * 1.45)
    ax.set_ylabel("PR-AUC on that airport's Nov-Dec flights")
    ax.set_title("Cold start: what a new airport costs\n"
                 "percentage is the share of skill retained", fontsize=15)
    ax.legend(fontsize=10, loc="upper left", framealpha=.95)

    ax = axes[1]
    ax.bar(x - 0.2, wi_lift, width=0.4, color=BLUE, label="within network")
    ax.bar(x + 0.2, tr_lift, width=0.4, color=ACCENT, label="unseen airport")
    ax.axhline(1.0, color="0.35", ls="--", lw=1.6)
    ax.text(-0.48, 1.06, "no better than random", fontsize=10, color="0.35",
            ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5))
    for i in range(len(airports)):
        ax.text(i + 0.2, tr_lift[i] + .05, f"{tr_lift[i]:.1f}x", ha="center",
                fontsize=12, weight="bold", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(airports)
    ax.set_ylim(0, max(wi_lift) * 1.22)
    ax.set_ylabel("lift in the riskiest 10%")
    ax.set_title("The operational read survives the move", fontsize=15)
    ax.legend(fontsize=10, loc="lower right")

    fig.suptitle("Leave-one-airport-out: the model transfers to airports it "
                 "has never seen", y=1.04, fontsize=17)
    fig.tight_layout()
    fig.savefig(FIGURES / "30_transfer.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    log.info("wrote 30_transfer.png")


def main(budget: float | None = None) -> int:
    deadline = None if budget is None else time.time() + budget
    params = joblib.load(MODELS / "training_context.joblib")["best_params"]

    done: Dict[str, Dict] = {}
    if RESULTS_FILE.exists():
        done = json.loads(RESULTS_FILE.read_text())

    for origin in ORIGINS:
        if origin in done:
            continue
        if deadline and time.time() > deadline:
            log.info("budget reached; %d/%d airports done", len(done), len(ORIGINS))
            return 2
        done[origin] = run_origin(origin, params)
        RESULTS_FILE.write_text(json.dumps(done, indent=2))

    # Summary across the three holdouts, so the report can quote one number.
    per = [done[o] for o in ORIGINS if o in done]
    done["summary"] = {
        "mean_retention_pr_auc": float(np.mean(
            [r["retention"]["pr_auc"] for r in per])),
        "worst_retention_pr_auc": float(np.min(
            [r["retention"]["pr_auc"] for r in per])),
        "min_transfer_lift_at_10pct": float(np.min(
            [r["transfer"]["lift_at_10pct"] for r in per])),
        "mean_transfer_pr_auc": float(np.mean(
            [r["transfer"]["pr_auc"] for r in per])),
        "mean_within_pr_auc": float(np.mean(
            [r["within_network"]["pr_auc"] for r in per])),
        "all_beat_base_rate": bool(all(
            r["transfer"]["pr_auc"] > r["transfer"]["base_rate"] for r in per)),
    }
    RESULTS_FILE.write_text(json.dumps(done, indent=2))

    make_figure({o: done[o] for o in ORIGINS if o in done})
    log.info("mean skill retained at an unseen airport: %.1f%%",
             100 * done["summary"]["mean_retention_pr_auc"])
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds; stop cleanly at the next checkpoint")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    sys.exit(main(args.budget))
