"""How far ahead can this actually be predicted?

The main model uses the weather observation at the scheduled departure hour.
That is legitimate — at the moment of scheduled push-back the current
observation exists — but it means the reported score is for a zero-hour
horizon. Since the ablation showed weather carries 0.141 of the 0.507 PR-AUC,
"we would need a forecast to go further ahead" is not a caveat that can be left
unquantified.

So quantify it. For a horizon of *h* hours, give each flight the observation
from *h* hours before its scheduled departure and retrain from scratch. That is
a **persistence forecast**: the crudest possible one, "conditions will be what
they are now". Real numerical weather prediction beats persistence at every
horizon beyond an hour or two, so the resulting curve is a **lower bound** on
what a forecast-fed model would achieve — the honest direction to be wrong in.

    python -m src.horizon --budget 35     # resumable
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
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score

from src import features as F
from src import models as M
from src.config import FIGURES, METRICS, MODE_A, MODELS
from src.data_loader import load_tables

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
ACCENT = "#c44e52"
BLUE = "#3b6978"

RESULTS_FILE = METRICS / "horizon.json"
HORIZONS = [0, 1, 2, 3, 6, 12, 24]


def run_horizon(tables, lag: int, params: Dict) -> Dict:
    df = F.build_feature_frame(tables, weather_lag_hours=lag)
    train, valid, test = F.temporal_split(df)
    train, (valid, test), enc = F.add_target_encodings(train, [valid, test])
    cols = F.feature_columns(train, MODE_A, enc)

    train = train.sort_values("sched_dep_utc", kind="mergesort")
    valid = valid.sort_values("sched_dep_utc", kind="mergesort")
    Xtr = F.as_model_frame(train, cols)
    Xva = F.as_model_frame(valid, cols)
    Xte = F.as_model_frame(test, cols)
    ytr = train["is_delayed"].to_numpy()
    yva = valid["is_delayed"].to_numpy()
    yte = test["is_delayed"].to_numpy()

    model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    p = model.predict_proba(Xte)[:, 1]

    n_k = int(0.10 * len(p))
    top = np.argsort(-p)[:n_k]
    return {
        "lag_hours": lag,
        "pr_auc": float(average_precision_score(yte, p)),
        "roc_auc": float(roc_auc_score(yte, p)),
        "precision_at_10pct": float(yte[top].mean()),
        "trees": int(model.best_iteration_),
    }


def main(budget: float | None) -> int:
    deadline = None if budget is None else time.time() + budget
    params = joblib.load(MODELS / "training_context.joblib")["best_params"]
    tables = load_tables()
    done: Dict[str, Dict] = {}
    if RESULTS_FILE.exists():
        done = json.loads(RESULTS_FILE.read_text())

    for lag in HORIZONS:
        key = str(lag)
        if key in done:
            continue
        if deadline and time.time() > deadline:
            log.info("budget reached; %d/%d horizons done", len(done), len(HORIZONS))
            return 2
        t0 = time.time()
        done[key] = run_horizon(tables, lag, params)
        RESULTS_FILE.write_text(json.dumps(done, indent=2))
        log.info("h=%2d  PR-AUC %.4f  ROC-AUC %.4f  prec@10%% %.3f  (%.0fs)",
                 lag, done[key]["pr_auc"], done[key]["roc_auc"],
                 done[key]["precision_at_10pct"], time.time() - t0)

    make_figure(done)
    return 0


def make_figure(done: Dict) -> None:
    lags = sorted(int(k) for k in done)
    pr = [done[str(l)]["pr_auc"] for l in lags]
    roc = [done[str(l)]["roc_auc"] for l in lags]
    prec = [done[str(l)]["precision_at_10pct"] for l in lags]

    # Reference points from the ablation: what the model scores with no weather
    # at all, and the base rate.
    no_weather = None
    abl_path = METRICS / "ablation.json"
    if abl_path.exists():
        no_weather = json.loads(abl_path.read_text())["weather"]["test_pr_auc"]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    ax = axes[0]
    ax.plot(lags, pr, marker="o", lw=2.8, color=ACCENT, label="PR-AUC")
    if no_weather is not None:
        ax.axhline(no_weather, ls="--", color="grey", lw=2)
        ax.text(0.3, no_weather + .004, f"no weather at all ({no_weather:.3f})",
                fontsize=11, color="#555")
    ax.axhline(0.250, ls=":", color="black", lw=1.5)
    ax.text(0.3, 0.256, "base rate (0.250)", fontsize=11)
    for l, v in zip(lags, pr):
        ax.annotate(f"{v:.3f}", (l, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10)
    ax.set_xlabel("forecast horizon (hours before scheduled departure)")
    ax.set_ylabel("test PR-AUC")
    ax.set_title("Persistence-forecast horizon curve")
    ax.legend(fontsize=11)

    ax = axes[1]
    ax.plot(lags, roc, marker="o", lw=2.6, color=BLUE, label="ROC-AUC")
    ax.plot(lags, prec, marker="s", lw=2.6, color=ACCENT,
            label="precision in riskiest 10%")
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_title("Ranking quality decays slowly")
    ax.legend(fontsize=11)

    fig.suptitle("Weather at h hours' notice: a lower bound on forecast-fed "
                 "performance", y=1.03, fontsize=17)
    fig.tight_layout()
    fig.savefig(FIGURES / "25_forecast_horizon.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote 25_forecast_horizon.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    sys.exit(main(args.budget))
