"""Severity: how bad, not just whether bad.

The first attempt at this was an L1 regressor on ``arr_delay`` over all
flights. It beat a constant-median predictor by 2.2% MAE, which is close to
nothing, and for a defensible reason: three quarters of flights are on time and
L1 loss is minimised by predicting the median, so it does. That model answered
a question ("what is the expected delay of a randomly chosen flight") that no
operations desk asks.

Three replacements, each matched to a decision that actually gets made:

``quantile``
    Quantile regression at the 50th and 90th percentiles. The desk does not
    want an expectation, it wants a reasonable worst case: "if this goes badly,
    how badly?" Scored with pinball loss against a constant-quantile baseline,
    plus empirical coverage of the P90 band.

``tiers``
    Separate classifiers for delay > 15, > 60 and > 120 minutes. Those
    thresholds are operationally distinct -- 15 min is the on-time metric,
    60 min breaks most connections, 120 min enters compensation territory.
    Classification is where the signal lives, so this buys severity resolution
    without giving up the thing the model is good at.

``conditional``
    Magnitude given lateness: trained only on flights that did arrive late.
    Combined with P(late) it decomposes expected disruption into probability
    times severity, which is how the two halves are actually used.

    python -m src.severity --step all --budget 35     # resumable
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
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from src import models as M
from src.config import FIGURES, METRICS, MODE_A, MODELS, N_JOBS, SEED
from src.pipeline import load_splits, xy

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
ACCENT = "#c44e52"
BLUE = "#3b6978"
GREEN = "#55a868"

RESULTS_FILE = METRICS / "severity_v2.json"

QUANTILES = [0.5, 0.9]
TIERS = [15, 60, 120]


def _load(default):
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return default


def _save(d: Dict) -> None:
    RESULTS_FILE.write_text(json.dumps(d, indent=2))


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Standard check loss. Lower is better; it is the loss quantile
    regression actually optimises, so it is the honest scoring rule here."""
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# ---------------------------------------------------------------------------


def step_quantile(train, valid, test, feats, params, results, deadline) -> bool:
    ytr = train["arr_delay"].to_numpy()
    yva = valid["arr_delay"].to_numpy()
    yte = test["arr_delay"].to_numpy()
    Xtr, _ = xy(train, feats)
    Xva, _ = xy(valid, feats)
    Xte, _ = xy(test, feats)

    block = results.setdefault("quantile", {})
    for q in QUANTILES:
        key = f"p{int(q * 100)}"
        if key in block:
            continue
        if deadline and time.time() > deadline:
            return False
        model = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                  n_estimators=3000, bagging_freq=1,
                                  n_jobs=N_JOBS, random_state=SEED,
                                  verbose=-1, **params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="quantile",
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        pred = model.predict(Xte)
        # Baseline: the same quantile of the training distribution, constant.
        const = float(np.quantile(ytr, q))
        block[key] = {
            "pinball": pinball_loss(yte, pred, q),
            "pinball_constant_baseline": pinball_loss(yte, np.full_like(yte, const,
                                                                        dtype=float), q),
            "constant_baseline_value_min": const,
            "coverage": float(np.mean(yte <= pred)),
            "coverage_target": q,
            "coverage_constant_baseline": float(np.mean(yte <= const)),
            "median_prediction_min": float(np.median(pred)),
            "trees": int(model.best_iteration_),
        }
        block[key]["pinball_improvement_pct"] = 100 * (
            1 - block[key]["pinball"] / block[key]["pinball_constant_baseline"])
        joblib.dump(model, MODELS / f"lightgbm_quantile_{key}.joblib")
        _save(results)
        log.info("quantile %s: pinball %.3f vs %.3f baseline (%.1f%% better), "
                 "coverage %.3f", key, block[key]["pinball"],
                 block[key]["pinball_constant_baseline"],
                 block[key]["pinball_improvement_pct"], block[key]["coverage"])
    return True


def step_tiers(train, valid, test, feats, params, results, deadline) -> bool:
    Xtr, _ = xy(train, feats)
    Xva, _ = xy(valid, feats)
    Xte, _ = xy(test, feats)

    block = results.setdefault("tiers", {})
    for t in TIERS:
        key = f"gt{t}"
        if key in block:
            continue
        if deadline and time.time() > deadline:
            return False
        ytr = (train["arr_delay"] > t).astype(int).to_numpy()
        yva = (valid["arr_delay"] > t).astype(int).to_numpy()
        yte = (test["arr_delay"] > t).astype(int).to_numpy()

        model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        p = model.predict_proba(Xte)[:, 1]

        # Operational read: take the riskiest 10% and see what fraction really
        # breached this tier.
        n_k = int(0.10 * len(p))
        top = np.argsort(-p)[:n_k]
        base = float(yte.mean())
        block[key] = {
            "threshold_min": t,
            "base_rate": base,
            "pr_auc": float(average_precision_score(yte, p)),
            "roc_auc": float(roc_auc_score(yte, p)),
            "pr_auc_lift_over_base": float(average_precision_score(yte, p)) / base,
            "precision_at_10pct": float(yte[top].mean()),
            "lift_at_10pct": float(yte[top].mean()) / base,
            "trees": int(model.best_iteration_),
        }
        joblib.dump(model, MODELS / f"lightgbm_tier_{key}.joblib")
        _save(results)
        log.info("tier >%d min: base %.3f  PR-AUC %.4f  ROC-AUC %.4f  "
                 "prec@10%% %.3f (lift %.2fx)", t, base, block[key]["pr_auc"],
                 block[key]["roc_auc"], block[key]["precision_at_10pct"],
                 block[key]["lift_at_10pct"])
    return True


def step_conditional(train, valid, test, feats, params, results, deadline) -> bool:
    """Magnitude given lateness. Trained only on flights that were late."""
    if "conditional" in results:
        return True
    if deadline and time.time() > deadline:
        return False

    tr_l = train[train["is_delayed"] == 1]
    va_l = valid[valid["is_delayed"] == 1]
    te_l = test[test["is_delayed"] == 1]
    Xtr, _ = xy(tr_l, feats)
    Xva, _ = xy(va_l, feats)
    Xte, _ = xy(te_l, feats)
    ytr = tr_l["arr_delay"].to_numpy()
    yte = te_l["arr_delay"].to_numpy()

    model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=3000,
                              bagging_freq=1, n_jobs=N_JOBS, random_state=SEED,
                              verbose=-1, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, va_l["arr_delay"].to_numpy())],
              eval_metric="l1", callbacks=[lgb.early_stopping(150, verbose=False)])
    pred = model.predict(Xte)
    const = float(np.median(ytr))

    from scipy.stats import spearmanr
    results["conditional"] = {
        "n_train_late": int(len(tr_l)),
        "n_test_late": int(len(te_l)),
        "mae_min": float(mean_absolute_error(yte, pred)),
        "mae_constant_baseline_min": float(
            mean_absolute_error(yte, np.full_like(yte, const, dtype=float))),
        "constant_baseline_min": const,
        "spearman_rho": float(spearmanr(pred, yte).statistic),
        "trees": int(model.best_iteration_),
    }
    results["conditional"]["improvement_pct"] = 100 * (
        1 - results["conditional"]["mae_min"]
        / results["conditional"]["mae_constant_baseline_min"])
    joblib.dump(model, MODELS / "lightgbm_conditional_delay.joblib")
    _save(results)
    log.info("conditional-on-late: MAE %.2f vs %.2f baseline (%.1f%% better)",
             results["conditional"]["mae_min"],
             results["conditional"]["mae_constant_baseline_min"],
             results["conditional"]["improvement_pct"])
    return True


# ---------------------------------------------------------------------------


def make_figures(results: Dict, train, test, feats) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- tiers -------------------------------------------------------
    ax = axes[0]
    tiers = results["tiers"]
    keys = [f"gt{t}" for t in TIERS]
    x = np.arange(len(keys))
    base = [tiers[k]["base_rate"] for k in keys]
    prec = [tiers[k]["precision_at_10pct"] for k in keys]
    ax.bar(x - 0.2, base, width=0.4, color=BLUE, label="base rate")
    ax.bar(x + 0.2, prec, width=0.4, color=ACCENT, label="precision in riskiest 10%")
    for i, k in enumerate(keys):
        ax.text(i + 0.2, prec[i] + .01, f"{tiers[k]['lift_at_10pct']:.1f}x",
                ha="center", fontsize=12, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"> {t} min" for t in TIERS])
    ax.set_ylabel("share of flights")
    ax.set_title("Severity tiers: rarer means higher lift")
    ax.legend(fontsize=10)

    ax = axes[1]
    aucs = [tiers[k]["roc_auc"] for k in keys]
    prs = [tiers[k]["pr_auc"] for k in keys]
    ax.plot(x, aucs, marker="o", lw=2.5, color=BLUE, label="ROC-AUC")
    ax.plot(x, prs, marker="s", lw=2.5, color=ACCENT, label="PR-AUC")
    ax.plot(x, base, marker="^", lw=1.8, ls="--", color="grey", label="base rate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"> {t} min" for t in TIERS])
    ax.set_ylim(0, 1)
    ax.set_title("Discrimination holds as the tier gets rarer")
    ax.legend(fontsize=10)

    # --- quantile coverage -------------------------------------------
    ax = axes[2]
    q = results["quantile"]
    labels, model_cov, base_cov, targets = [], [], [], []
    for qq in QUANTILES:
        k = f"p{int(qq * 100)}"
        labels.append(k.upper())
        model_cov.append(q[k]["coverage"])
        base_cov.append(q[k]["coverage_constant_baseline"])
        targets.append(qq)
    xx = np.arange(len(labels))
    ax.bar(xx - 0.2, model_cov, width=0.4, color=ACCENT, label="quantile model")
    ax.bar(xx + 0.2, base_cov, width=0.4, color=BLUE, label="constant baseline")
    for i, t in enumerate(targets):
        ax.hlines(t, i - 0.45, i + 0.45, color="black", lw=2.2, ls=":")
        ax.text(i, t + .02, f"target {t:.0%}", ha="center", fontsize=11)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of flights at or below the prediction")
    ax.set_title("Quantile calibration on the test period")
    ax.legend(fontsize=10, loc="lower right")

    fig.suptitle("Severity, reframed: tiers and quantiles instead of a "
                 "conditional mean", y=1.03, fontsize=17)
    fig.tight_layout()
    fig.savefig(FIGURES / "23_severity_tiers_quantiles.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- the P90 band, visually --------------------------------------
    p50 = joblib.load(MODELS / "lightgbm_quantile_p50.joblib")
    p90 = joblib.load(MODELS / "lightgbm_quantile_p90.joblib")
    Xte, _ = xy(test, feats)
    a = test["arr_delay"].to_numpy()
    pred50, pred90 = p50.predict(Xte), p90.predict(Xte)

    order = np.argsort(pred90)
    n_bin = 40
    bins = np.array_split(order, n_bin)
    xs = np.arange(n_bin)
    m50 = [np.median(pred50[b]) for b in bins]
    m90 = [np.median(pred90[b]) for b in bins]
    act50 = [np.median(a[b]) for b in bins]
    act90 = [np.quantile(a[b], 0.9) for b in bins]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(xs, m90, lw=2.6, color=ACCENT, label="predicted P90")
    ax.plot(xs, act90, lw=2.0, ls="--", color=ACCENT, alpha=.65,
            label="actual P90 within bin")
    ax.plot(xs, m50, lw=2.6, color=BLUE, label="predicted P50")
    ax.plot(xs, act50, lw=2.0, ls="--", color=BLUE, alpha=.65,
            label="actual median within bin")
    ax.fill_between(xs, m50, m90, color=ACCENT, alpha=.10)
    ax.axhline(15, color="grey", ls=":", lw=1.5)
    ax.text(0.5, 17, "on-time threshold", fontsize=10, color="grey")
    ax.set_xlabel("flights ordered by predicted P90 (40 equal bins)")
    ax.set_ylabel("arrival delay (minutes)")
    ax.set_title("The P90 band tracks the real upper tail\n"
                 "shaded region is the model's P50-P90 range", fontsize=15)
    ax.legend(fontsize=11)
    fig.savefig(FIGURES / "24_severity_p90_band.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote figures 23-24")


# ---------------------------------------------------------------------------


def main(step: str, budget: float | None) -> int:
    deadline = None if budget is None else time.time() + budget
    train, valid, test, manifest = load_splits()
    feats: List[str] = manifest["features"][MODE_A]
    params = joblib.load(MODELS / "training_context.joblib")["best_params"]
    results = _load({})

    todo = ["quantile", "tiers", "conditional"] if step == "all" else [step]
    for s in todo:
        fn = {"quantile": step_quantile, "tiers": step_tiers,
              "conditional": step_conditional}[s]
        if not fn(train, valid, test, feats, params, results, deadline):
            log.info("budget reached during '%s' -- rerun to continue", s)
            return 2

    if all(k in results for k in ("quantile", "tiers", "conditional")):
        make_figures(results, train, test, feats)
    _save(results)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["all", "quantile", "tiers", "conditional"])
    ap.add_argument("--budget", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    sys.exit(main(args.step, args.budget))
