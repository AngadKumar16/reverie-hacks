"""Evaluation on the held-out Nov-Dec period.

Covers four things the report needs:

1. **Discrimination** -- ROC-AUC and PR-AUC for every model, against the
   baselines. PR-AUC is the headline metric: with a 25% positive rate and an
   operations desk that can only act on a small slice of flights, the
   precision-recall trade-off is what matters, not the ROC curve.
2. **Calibration** -- reliability curves and Brier score. A probability that
   says "40%" has to mean 40%, otherwise the cost-based threshold below is
   meaningless. We also measure how much calibration *drifts* from the
   validation period into the test period, and whether a small rolling
   recalibration recovers it.
3. **Decision quality** -- the cost-optimal threshold, precision@k for a fixed
   operational capacity, and lift over the base rate.
4. **Where it fails** -- error broken out by month, carrier, origin, hour,
   distance and weather so the limitations section is evidence-based rather
   than boilerplate.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    CAPACITY_FRACTION,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    FIGURES,
    METRICS,
    MODE_A,
    MODE_B,
    MODELS,
)
from src.features import CATEGORICAL_FEATURES
from src.pipeline import load_splits, xy

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
ACCENT = "#c44e52"
BLUE = "#3b6978"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MODEL_LABELS = {
    "prior": "Base rate (no model)",
    "historical_rate": "Historical-rate rule",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost (tuned)",
    "lightgbm": "LightGBM (tuned)",
    "lightgbm_cal": "LightGBM + isotonic",
    "lightgbm_gate": "LightGBM, post-push-back",
}


def _save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    log.info("wrote %s.png", name)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def predict(name: str, model, X: pd.DataFrame) -> np.ndarray:
    if name == "xgboost":
        cols = joblib.load(MODELS / "xgboost_columns.joblib")
        Xn = pd.get_dummies(X, columns=[c for c in CATEGORICAL_FEATURES
                                        if c in X.columns], dummy_na=True)
        Xn = Xn.reindex(columns=cols, fill_value=0)
        return model.predict_proba(Xn)[:, 1]
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------


def discrimination(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }


def expected_cost(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return (fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE) / len(y)


def sweep_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[pd.DataFrame, float, float]:
    grid = np.linspace(0.02, 0.95, 187)
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": f1_score(y, pred, zero_division=0),
            "alert_rate": pred.mean(),
            "cost_per_flight": (fp * COST_FALSE_POSITIVE
                                + fn * COST_FALSE_NEGATIVE) / len(y),
        })
    df = pd.DataFrame(rows)
    return df, float(df.loc[df["cost_per_flight"].idxmin(), "threshold"]), \
        float(df.loc[df["f1"].idxmax(), "threshold"])


def precision_at_capacity(y: np.ndarray, p: np.ndarray,
                          k: float = CAPACITY_FRACTION) -> Dict[str, float]:
    n = max(int(len(p) * k), 1)
    order = np.argsort(-p)[:n]
    prec = float(y[order].mean())
    base = float(y.mean())
    return {
        "k": k,
        "n_flights_flagged": n,
        "precision_at_k": prec,
        "recall_at_k": float(y[order].sum() / y.sum()),
        "lift": prec / base if base else float("nan"),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_curves(y: np.ndarray, preds: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    order = sorted(preds, key=lambda k: -average_precision_score(y, preds[k]))
    colors = sns.color_palette("mako", len(order))

    for c, name in zip(colors, order):
        p = preds[name]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=2.2, color=c,
                     label=f"{MODEL_LABELS.get(name, name)} ({roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, lw=2.2, color=c,
                     label=f"{MODEL_LABELS.get(name, name)} ({average_precision_score(y, p):.3f})")

    axes[0].plot([0, 1], [0, 1], ls=":", color="grey")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(fontsize=10, loc="lower right")

    axes[1].axhline(y.mean(), ls=":", color="grey",
                    label=f"base rate ({y.mean():.3f})")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall")
    axes[1].legend(fontsize=10, loc="upper right")
    fig.suptitle("Held-out test period (Nov-Dec 2013), pre-flight features only",
                 y=1.02, fontsize=17)
    _save(fig, "08_roc_pr_curves")


def fig_calibration(y_va, p_va, y_te, p_te, p_te_cal) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3))

    ax = axes[0]
    for label, (yy, pp), col in [
        ("validation (Sep-Oct), uncalibrated", (y_va, p_va), BLUE),
        ("test (Nov-Dec), uncalibrated", (y_te, p_te), ACCENT),
        ("test (Nov-Dec), isotonic on validation", (y_te, p_te_cal), "#55a868"),
    ]:
        frac, mean_pred = calibration_curve(yy, pp, n_bins=15, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", lw=2.2, color=col,
                label=f"{label}  (Brier {brier_score_loss(yy, pp):.4f})")
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability")
    ax.legend(fontsize=10, loc="upper left")

    ax = axes[1]
    ax.hist(p_te, bins=40, alpha=.75, color=BLUE, label="test predictions")
    ax.axvline(p_te.mean(), color=BLUE, ls="--",
               label=f"mean prediction {p_te.mean():.3f}")
    ax.axvline(y_te.mean(), color=ACCENT, ls="--",
               label=f"actual test rate {y_te.mean():.3f}")
    ax.set_xlabel("predicted probability of arriving >15 min late")
    ax.set_ylabel("flights")
    ax.set_title("Prediction distribution")
    ax.legend(fontsize=10)
    fig.suptitle("Calibration drifts when the base rate shifts between periods",
                 y=1.02, fontsize=17)
    _save(fig, "09_calibration")


def fig_threshold(sweep: pd.DataFrame, t_cost: float, t_f1: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(sweep["threshold"], sweep["precision"], label="precision", lw=2.4, color=BLUE)
    ax.plot(sweep["threshold"], sweep["recall"], label="recall", lw=2.4, color=ACCENT)
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", lw=2.4, color="#55a868")
    ax.plot(sweep["threshold"], sweep["alert_rate"], label="share of flights flagged",
            lw=1.8, ls="--", color="#8172b3")
    ax.axvline(t_cost, color="black", ls=":", lw=2)
    ax.annotate(f"cost-optimal\nt = {t_cost:.2f}", (t_cost, .93),
                xytext=(8, 0), textcoords="offset points", fontsize=11)
    ax.set_xlabel("decision threshold")
    ax.set_title("Operating characteristics")
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(sweep["threshold"], sweep["cost_per_flight"], lw=2.6, color=BLUE)
    ax.axvline(t_cost, color="black", ls=":", lw=2)
    best = sweep["cost_per_flight"].min()
    ax.scatter([t_cost], [best], color=ACCENT, zorder=5, s=70)
    ax.annotate(f"min cost {best:.3f} / flight at t = {t_cost:.2f}",
                (t_cost, best), xytext=(12, 20), textcoords="offset points",
                fontsize=11, arrowprops=dict(arrowstyle="->", color="grey"))
    ax.set_xlabel("decision threshold")
    ax.set_ylabel(f"expected cost per flight\n(FN = {COST_FALSE_NEGATIVE:g}x FP)")
    ax.set_title("Cost-sensitive threshold selection")
    fig.suptitle("The 0.5 default is the wrong threshold for this decision",
                 y=1.02, fontsize=17)
    _save(fig, "10_threshold_selection")


def fig_confusion(y: np.ndarray, p: np.ndarray, t: float) -> None:
    cm = confusion_matrix(y, (p >= t).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, norm, title in [
        (axes[0], None, "counts"),
        (axes[1], "true", "row-normalised (recall view)"),
    ]:
        data = cm if norm is None else cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(data, annot=True, fmt=",d" if norm is None else ".2%",
                    cmap="mako_r", cbar=False, ax=ax,
                    xticklabels=["predicted on-time", "predicted late"],
                    yticklabels=["actually on-time", "actually late"])
        ax.set_title(title)
    fig.suptitle(f"Confusion matrix at the cost-optimal threshold t = {t:.2f} "
                 f"(test period)", y=1.03, fontsize=16)
    _save(fig, "11_confusion_matrix")


def fig_segment_errors(test: pd.DataFrame, p: np.ndarray, t: float) -> pd.DataFrame:
    d = test.copy()
    d["p"] = p
    d["pred"] = (p >= t).astype(int)

    def block(keys, label):
        g = d.groupby(keys, observed=True)
        out = pd.DataFrame({
            "n": g.size(),
            "actual_rate": g["is_delayed"].mean(),
            "mean_pred": g["p"].mean(),
            "roc_auc": g.apply(
                lambda s: roc_auc_score(s["is_delayed"], s["p"])
                if s["is_delayed"].nunique() > 1 else np.nan,
                include_groups=False),
        })
        out["calibration_gap"] = out["mean_pred"] - out["actual_rate"]
        out["segment"] = label
        return out.reset_index().rename(columns={keys[0]: "level"})

    d["distance_band"] = pd.cut(d["distance"], [0, 500, 1000, 1800, 5000],
                                labels=["<500 mi", "500-1000", "1000-1800", "1800+"])
    d["weather_band"] = np.where(d["wx_precip"] > 0, "precipitating", "dry")
    d["hour_band"] = pd.cut(d["sched_dep_hour"], [-1, 8, 12, 17, 24],
                            labels=["early (<9)", "morning", "afternoon", "evening"])

    frames = [block(["month"], "month"), block(["origin"], "origin"),
              block(["carrier"], "carrier"), block(["distance_band"], "distance"),
              block(["weather_band"], "weather"), block(["hour_band"], "time of day")]
    seg = pd.concat(frames, ignore_index=True)
    seg = seg[seg["n"] >= 200]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for ax, label in zip(axes.ravel(), ["month", "origin", "carrier",
                                        "distance", "weather", "time of day"]):
        sub = seg[seg["segment"] == label].copy()
        if label == "month":
            sub["level"] = sub["level"].map(lambda m: MONTHS[int(m) - 1])
        sub = sub.sort_values("actual_rate")
        x = np.arange(len(sub))
        ax.bar(x - .2, sub["actual_rate"], width=.4, label="actual late rate",
               color=ACCENT)
        ax.bar(x + .2, sub["mean_pred"], width=.4, label="mean prediction",
               color=BLUE)
        ax2 = ax.twinx()
        ax2.plot(x, sub["roc_auc"], marker="o", color="black", lw=1.6,
                 label="ROC-AUC within segment")
        ax2.set_ylim(.45, .85)
        ax2.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["level"].astype(str), rotation=45, ha="right",
                           fontsize=10)
        ax.set_title(label)
        ax.set_ylim(0, .62)
        if label == "month":
            ax.legend(fontsize=9, loc="upper left")
            ax2.legend(fontsize=9, loc="upper right")
    fig.suptitle("Where the model is well-calibrated and where it is not "
                 "(test period, segments with n >= 200)", y=1.0, fontsize=17)
    fig.tight_layout()
    _save(fig, "12_segment_error_analysis")
    return seg


def fig_mode_comparison(y, p_pre, p_gate) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for label, p, col in [("pre-flight (deployable)", p_pre, BLUE),
                          ("post-push-back (adds observed dep_delay)", p_gate, ACCENT)]:
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, lw=2.6, color=col,
                label=f"{label}\n  PR-AUC {average_precision_score(y, p):.3f} | "
                      f"ROC-AUC {roc_auc_score(y, p):.3f}")
    ax.axhline(y.mean(), ls=":", color="grey", label=f"base rate {y.mean():.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("What the departure signal is worth\n"
                 "Same model, same data, two decision points", fontsize=15)
    ax.legend(fontsize=10, loc="lower left")
    _save(fig, "13_prediction_horizon")


def fig_severity(y_min: np.ndarray, pred_min: np.ndarray, baseline: float) -> Dict:
    mae = float(mean_absolute_error(y_min, pred_min))
    mae_base = float(mean_absolute_error(y_min, np.full_like(y_min, baseline,
                                                             dtype=float)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sample = np.random.default_rng(0).choice(len(y_min), size=min(6000, len(y_min)),
                                             replace=False)
    axes[0].scatter(pred_min[sample], np.clip(y_min[sample], -60, 300), s=6,
                    alpha=.18, color=BLUE)
    lims = [-40, 200]
    axes[0].plot(lims, lims, ls="--", color=ACCENT, lw=2)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(-60, 300)
    axes[0].set_xlabel("predicted delay (minutes)")
    axes[0].set_ylabel("actual delay (minutes, clipped)")
    axes[0].set_title(f"Predicted vs actual\nMAE {mae:.1f} min "
                      f"(median baseline {mae_base:.1f} min)")

    bands = pd.qcut(pred_min, 10, duplicates="drop")
    dfb = pd.DataFrame({"band": bands, "actual": y_min, "pred": pred_min})
    g = dfb.groupby("band", observed=True).agg(
        actual=("actual", "median"), pred=("pred", "median"), n=("actual", "size"))
    x = np.arange(len(g))
    axes[1].plot(x, g["pred"], marker="o", lw=2.4, color=BLUE, label="predicted (median)")
    axes[1].plot(x, g["actual"], marker="s", lw=2.4, color=ACCENT, label="actual (median)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"D{i+1}" for i in x])
    axes[1].set_xlabel("decile of predicted delay")
    axes[1].set_ylabel("arrival delay (minutes)")
    axes[1].set_title("Ranking quality by decile")
    axes[1].legend(fontsize=11)
    fig.suptitle("Severity head: how late, not just whether late", y=1.02, fontsize=17)
    _save(fig, "14_severity_model")
    return {"mae_min": mae, "mae_median_baseline_min": mae_base,
            "improvement_pct": 100 * (1 - mae / mae_base)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    train, valid, test, manifest = load_splits()
    feats_a: List[str] = manifest["features"][MODE_A]
    feats_b: List[str] = manifest["features"][MODE_B]

    Xva, yva = xy(valid, feats_a)
    Xte, yte = xy(test, feats_a)

    names = ["prior", "historical_rate", "logistic_regression", "random_forest",
             "lightgbm"]
    if (MODELS / "xgboost.joblib").exists():
        names.insert(4, "xgboost")

    preds_te, preds_va, results = {}, {}, {}
    for name in names:
        model = joblib.load(MODELS / f"{name}.joblib")
        preds_va[name] = predict(name, model, Xva)
        preds_te[name] = predict(name, model, Xte)
        results[name] = {
            "valid": discrimination(yva, preds_va[name]),
            "test": discrimination(yte, preds_te[name]),
        }
        log.info("%-20s test PR-AUC %.4f  ROC-AUC %.4f",
                 name, results[name]["test"]["pr_auc"],
                 results[name]["test"]["roc_auc"])

    # ---- calibration -------------------------------------------------
    p_va, p_te = preds_va["lightgbm"], preds_te["lightgbm"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)
    p_te_cal = iso.predict(p_te)
    results["lightgbm_cal"] = {"test": discrimination(yte, p_te_cal)}

    # A realistic online update: refit the calibrator on the first 14 days of
    # the test period, then score the remainder. This is what a deployed system
    # would actually do once new labels arrive.
    test_sorted = test.sort_values("sched_dep_utc").reset_index(drop=True)
    Xte_s, yte_s = xy(test_sorted, feats_a)
    p_te_s = predict("lightgbm", joblib.load(MODELS / "lightgbm.joblib"), Xte_s)
    cutoff = test_sorted["sched_dep_utc"].min() + pd.Timedelta(days=14)
    warm = (test_sorted["sched_dep_utc"] < cutoff).to_numpy()
    iso_online = IsotonicRegression(out_of_bounds="clip").fit(p_te_s[warm], yte_s[warm])
    p_rest_cal = iso_online.predict(p_te_s[~warm])
    results["rolling_recalibration"] = {
        "n_warmup_flights": int(warm.sum()),
        "brier_uncalibrated": float(brier_score_loss(yte_s[~warm], p_te_s[~warm])),
        "brier_isotonic_from_validation": float(
            brier_score_loss(yte_s[~warm], iso.predict(p_te_s[~warm]))),
        "brier_isotonic_rolling": float(brier_score_loss(yte_s[~warm], p_rest_cal)),
    }

    # ---- threshold & decisions ---------------------------------------
    sweep_va, t_cost, t_f1 = sweep_threshold(yva, iso.predict(p_va))
    sweep_te, t_cost_te, _ = sweep_threshold(yte, p_te_cal)
    at_t = sweep_te.iloc[(sweep_te["threshold"] - t_cost).abs().argsort().iloc[0]]
    results["decision"] = {
        "threshold_chosen_on_validation": t_cost,
        "threshold_f1_optimal_validation": t_f1,
        "threshold_oracle_on_test": t_cost_te,
        "cost_ratio_fn_to_fp": COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE,
        "test_at_chosen_threshold": {
            "precision": float(at_t["precision"]), "recall": float(at_t["recall"]),
            "f1": float(at_t["f1"]), "alert_rate": float(at_t["alert_rate"]),
            "cost_per_flight": float(at_t["cost_per_flight"]),
        },
        "cost_at_default_0.5": float(expected_cost(yte, p_te_cal, 0.5)),
        "cost_of_alerting_nothing": float(yte.mean() * COST_FALSE_NEGATIVE),
        "cost_of_alerting_everything": float((1 - yte.mean()) * COST_FALSE_POSITIVE),
    }
    results["capacity"] = precision_at_capacity(yte, p_te_cal)

    # ---- prediction horizon ------------------------------------------
    gate = joblib.load(MODELS / "lightgbm_gate.joblib")
    Xte_b, _ = xy(test, feats_b)
    p_gate = gate.predict_proba(Xte_b)[:, 1]
    results["lightgbm_gate"] = {"test": discrimination(yte, p_gate)}

    # ---- severity ------------------------------------------------------
    reg = joblib.load(MODELS / "lightgbm_severity.joblib")
    pred_min = reg.predict(Xte)
    results["severity"] = fig_severity(test["arr_delay"].to_numpy(), pred_min,
                                       float(train["arr_delay"].median()))

    # ---- figures -------------------------------------------------------
    fig_curves(yte, {**preds_te, "lightgbm_cal": p_te_cal})
    fig_calibration(yva, p_va, yte, p_te, p_te_cal)
    fig_threshold(sweep_va, t_cost, t_f1)
    fig_confusion(yte, p_te_cal, t_cost)
    seg = fig_segment_errors(test, p_te_cal, t_cost)
    fig_mode_comparison(yte, p_te_cal, p_gate)

    seg.to_csv(METRICS / "segment_errors.csv", index=False)
    sweep_va.to_csv(METRICS / "threshold_sweep_validation.csv", index=False)
    results["base_rates"] = {"valid": float(yva.mean()), "test": float(yte.mean())}
    (METRICS / "evaluation.json").write_text(json.dumps(results, indent=2))

    np.save(METRICS / "test_predictions.npy", p_te_cal)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    main()
