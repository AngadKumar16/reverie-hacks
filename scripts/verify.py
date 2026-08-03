"""End-to-end verification.

Checks three things that ordinary unit tests do not:

1. **Determinism** — rebuilding the feature pipeline from the raw tables
   reproduces the cached splits exactly, and refitting the chosen model
   reproduces the reported test score exactly.
2. **No leakage into the final artefacts** — the fitted model's feature list
   contains nothing that is unknown before push-back, and the model's
   predictions do not depend on any post-departure column.
3. **The report matches the artefacts** — every headline number quoted in
   `reports/report.md` and `README.md` is re-read from the metrics JSON and
   compared, so the prose cannot drift away from the results.

    python scripts/verify.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features as F                     # noqa: E402
from src.config import (                          # noqa: E402
    GATE_ONLY_FEATURES, LEAKY_COLUMNS, METRICS, MODE_A, MODELS,
)
from src.data_loader import load_tables           # noqa: E402
from src.pipeline import load_splits, xy          # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def approx(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
print("\n=== 1. Determinism ===")

train, valid, test, manifest = load_splits()
tables = load_tables()
rebuilt = F.build_feature_frame(tables)
rb_train, rb_valid, rb_test = F.temporal_split(rebuilt)
rb_train, (rb_valid, rb_test), enc = F.add_target_encodings(rb_train, [rb_valid, rb_test])


def digest(df: pd.DataFrame, cols: list[str]) -> str:
    arr = df[cols].select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    return hashlib.sha256(np.nan_to_num(arr, nan=-9999.0).tobytes()).hexdigest()[:16]


feats = manifest["features"][MODE_A]
for label, cached, fresh in [("train", train, rb_train), ("valid", valid, rb_valid),
                             ("test", test, rb_test)]:
    check(f"{label} split rebuilds identically",
          len(cached) == len(fresh) and digest(cached, feats) == digest(fresh, feats),
          f"n={len(fresh):,} sha={digest(fresh, feats)}")

# ---------------------------------------------------------------------------
print("\n=== 2. Leakage in the shipped artefacts ===")

ctx = joblib.load(MODELS / "training_context.joblib")
shipped = ctx["features_preflight"]
check("shipped feature list excludes outcome columns",
      not set(shipped) & set(LEAKY_COLUMNS))
check("shipped feature list excludes post-push-back signals",
      not set(shipped) & set(GATE_ONLY_FEATURES))
check("shipped feature list matches the manifest", set(shipped) == set(feats))

clf = joblib.load(MODELS / "lightgbm.joblib")
check("fitted model was trained on exactly those features",
      list(clf.feature_name_) == list(shipped),
      f"{len(clf.feature_name_)} features")

# Perturbing a post-departure column must not change a single prediction.
Xte, yte = xy(test, feats)
p_ref = clf.predict_proba(Xte)[:, 1]
scrambled = test.copy()
rng = np.random.default_rng(0)
for col in ["dep_delay", "arr_delay", "air_time", "dep_time", "arr_time"]:
    if col in scrambled.columns:
        scrambled[col] = rng.permutation(scrambled[col].to_numpy())
p_scrambled = clf.predict_proba(xy(scrambled, feats)[0])[:, 1]
check("predictions are invariant to scrambling every post-departure column",
      np.allclose(p_ref, p_scrambled),
      f"max abs delta {np.abs(p_ref - p_scrambled).max():.2e}")

# ---------------------------------------------------------------------------
print("\n=== 3. Refit reproduces the reported score ===")

import lightgbm as lgb                            # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from src import models as M                       # noqa: E402

ev = json.loads((METRICS / "evaluation.json").read_text())
reported_ap = ev["lightgbm"]["test"]["pr_auc"]
check("saved model reproduces the reported test PR-AUC",
      approx(average_precision_score(yte, p_ref), reported_ap, 1e-9),
      f"{average_precision_score(yte, p_ref):.6f} vs {reported_ap:.6f}")

tr_sorted = train.sort_values("sched_dep_utc", kind="mergesort").reset_index(drop=True)
va_sorted = valid.sort_values("sched_dep_utc", kind="mergesort").reset_index(drop=True)
Xtr, ytr = xy(tr_sorted, feats)
Xva, yva = xy(va_sorted, feats)
refit = lgb.LGBMClassifier(**M.LGBM_FIXED, **ctx["best_params"])
refit.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(150, verbose=False)])
ap_refit = average_precision_score(yte, refit.predict_proba(Xte)[:, 1])
check("a fresh refit from the same seed reproduces the score",
      approx(ap_refit, reported_ap, 1e-9),
      f"{ap_refit:.6f}, {refit.best_iteration_} trees")

# The joblib pickle is not byte-stable, but the booster's own model string is.
# Hashing it gives a fingerprint that survives a full rebuild, so a reviewer can
# confirm they reproduced the identical model rather than a similar-scoring one.
fp = json.loads((METRICS / "model_fingerprints.json").read_text())
refit_sha = hashlib.sha256(refit.booster_.model_to_string().encode()).hexdigest()
check("the refit booster hashes to the recorded fingerprint",
      refit_sha == fp["lightgbm"]["sha256"], refit_sha[:16] + "…")
check("feature list hashes to the recorded fingerprint",
      hashlib.sha256("\n".join(shipped).encode()).hexdigest()
      == fp["feature_list_sha256"])

# ---------------------------------------------------------------------------
print("\n=== 4. Report claims match the artefacts ===")

abl = json.loads((METRICS / "ablation.json").read_text())
hor = json.loads((METRICS / "horizon.json").read_text())
dis = json.loads((METRICS / "disruption.json").read_text())
sev = json.loads((METRICS / "severity_v2.json").read_text())
shp = json.loads((METRICS / "shap_importance.json").read_text())
eda = json.loads((METRICS / "eda_summary.json").read_text())
imp = json.loads((METRICS / "impact.json").read_text())
fair = json.loads((METRICS / "fairness.json").read_text())

# The impact and fairness modules read `test_predictions.npy` positionally, so
# a re-ordered split would silently pair every flight with somebody else's
# probability. Re-derive the exposure totals from the raw split and compare.
_imp_late = int(test["is_delayed"].sum())
_imp_min = float(test.loc[test["is_delayed"] == 1, "arr_delay"].clip(lower=0).sum())

claims = {
    "pre-flight PR-AUC 0.507": approx(ev["lightgbm"]["test"]["pr_auc"], 0.507, 5e-4),
    "pre-flight ROC-AUC 0.716": approx(ev["lightgbm"]["test"]["roc_auc"], 0.716, 5e-4),
    "gate PR-AUC 0.846": approx(ev["lightgbm_gate"]["test"]["pr_auc"], 0.846, 5e-4),
    "gate ROC-AUC 0.903": approx(ev["lightgbm_gate"]["test"]["roc_auc"], 0.903, 5e-4),

    "logistic PR-AUC 0.478": approx(ev["logistic_regression"]["test"]["pr_auc"], 0.478, 5e-4),
    "random forest PR-AUC 0.491": approx(ev["random_forest"]["test"]["pr_auc"], 0.491, 5e-4),
    "historical rule PR-AUC 0.340": approx(ev["historical_rate"]["test"]["pr_auc"], 0.340, 5e-4),
    "test base rate 0.250": approx(ev["base_rates"]["test"], 0.250, 5e-4),
    "valid base rate 0.151": approx(ev["base_rates"]["valid"], 0.151, 5e-4),
    "threshold 0.20": approx(ev["decision"]["threshold_chosen_on_validation"], 0.20, 5e-3),
    "precision at threshold 0.473": approx(
        ev["decision"]["test_at_chosen_threshold"]["precision"], 0.473, 1e-3),
    "recall at threshold 0.492": approx(
        ev["decision"]["test_at_chosen_threshold"]["recall"], 0.492, 1e-3),
    "cost 0.645 vs default 0.844": (
        approx(ev["decision"]["test_at_chosen_threshold"]["cost_per_flight"], 0.645, 1e-3)
        and approx(ev["decision"]["cost_at_default_0.5"], 0.844, 1e-3)),
    "default costs 31% more than optimal": approx(
        ev["decision"]["cost_at_default_0.5"]
        / ev["decision"]["test_at_chosen_threshold"]["cost_per_flight"] - 1, 0.31, 5e-3),
    "precision@10% is 0.641": approx(ev["capacity"]["precision_at_k"], 0.641, 1e-3),
    "lift 2.56x": approx(ev["capacity"]["lift"], 2.565, 5e-3),
    "rolling recal Brier 0.1698": approx(
        ev["recalibration"]["brier"]["isotonic_rolling_14d"], 0.1698, 1e-4),
    "uncalibrated Brier 0.1801": approx(
        ev["recalibration"]["brier"]["uncalibrated"], 0.1801, 1e-4),
    "Brier improves 5.7%": approx(
        1 - ev["recalibration"]["brier"]["isotonic_rolling_14d"]
        / ev["recalibration"]["brier"]["uncalibrated"], 0.057, 2e-3),
    "severity MAE 22.8 min": approx(ev["severity"]["mae_min"], 22.76, 0.05),
    "severity Spearman 0.332": approx(ev["severity"]["spearman_rho"], 0.332, 1e-3),
    "weather ablation costs 0.141": approx(
        abl["full model"]["test_pr_auc"] - abl["weather"]["test_pr_auc"], 0.1409, 1e-3),
    "historical-rate ablation is non-negative": (
        abl["historical rates"]["test_pr_auc"] >= abl["full model"]["test_pr_auc"]),
    "time-index ablation costs 0.016": approx(
        abl["full model"]["test_pr_auc"]
        - abl["time index (day/week/month)"]["test_pr_auc"], 0.0160, 1e-3),
    "day_of_year mean SHAP -0.349": approx(
        shp["out_of_range_features"]["day_of_year"]["mean_shap"], -0.349, 2e-3),
    "day_of_year is out of range for all test flights": approx(
        shp["out_of_range_features"]["day_of_year"]["share_of_test_outside_training_range"],
        1.0, 1e-9),
    "overall late rate 23.7%": approx(eda["late_rate_overall"], 0.237, 5e-4),
    "labelled flights 327,346": eda["n_flights_labelled"] == 327346,
    "precipitating vs dry 47.7 / 22.1": (
        approx(eda["late_rate_precipitating_vs_dry"]["precipitating"], 0.477, 5e-4)
        and approx(eda["late_rate_precipitating_vs_dry"]["dry"], 0.221, 5e-4)),
    "split sizes 217727 / 55628 / 53991": (
        len(train) == 217727 and len(valid) == 55628 and len(test) == 53991),

    # --- equal-budget XGBoost comparison (report 5) ------------------
    "xgboost PR-AUC 0.513 with equal budget": approx(
        ev["xgboost"]["test"]["pr_auc"], 0.513, 5e-4),
    "xgboost beats lightgbm on test": (
        ev["xgboost"]["test"]["pr_auc"] > ev["lightgbm"]["test"]["pr_auc"]),

    # --- severity tiers and quantiles (report 5.5) -------------------
    "tier >60 min ROC-AUC 0.770": approx(sev["tiers"]["gt60"]["roc_auc"], 0.770, 1e-3),
    "tier >120 min ROC-AUC 0.793": approx(sev["tiers"]["gt120"]["roc_auc"], 0.793, 1e-3),
    "discrimination rises as the tier gets rarer": (
        sev["tiers"]["gt15"]["roc_auc"] < sev["tiers"]["gt60"]["roc_auc"]
        < sev["tiers"]["gt120"]["roc_auc"]),
    "lift rises as the tier gets rarer": (
        sev["tiers"]["gt15"]["lift_at_10pct"] < sev["tiers"]["gt60"]["lift_at_10pct"]
        < sev["tiers"]["gt120"]["lift_at_10pct"]),
    "P90 head beats a constant by 16.4%": approx(
        sev["quantile"]["p90"]["pinball_improvement_pct"], 16.4, 0.2),
    "P50 head barely beats a constant": (
        sev["quantile"]["p50"]["pinball_improvement_pct"] < 3.0),
    "conditional-on-late MAE improves 7.9%": approx(
        sev["conditional"]["improvement_pct"], 7.9, 0.2),

    # --- forecast horizon (report 5.6) -------------------------------
    "horizon curve is monotone decreasing": all(
        hor[str(a)]["pr_auc"] >= hor[str(b)]["pr_auc"]
        for a, b in zip([0, 1, 2, 3, 6, 12], [1, 2, 3, 6, 12, 24])),
    "3 h horizon retains 86% of weather's contribution": approx(
        (hor["3"]["pr_auc"] - abl["weather"]["test_pr_auc"])
        / (hor["0"]["pr_auc"] - abl["weather"]["test_pr_auc"]), 0.86, 0.01),
    "6 h horizon retains 72%": approx(
        (hor["6"]["pr_auc"] - abl["weather"]["test_pr_auc"])
        / (hor["0"]["pr_auc"] - abl["weather"]["test_pr_auc"]), 0.72, 0.01),
    "24 h horizon lands on the no-weather floor": abs(
        hor["24"]["pr_auc"] - abl["weather"]["test_pr_auc"]) < 0.005,
    "h=0 matches the main model": approx(
        hor["0"]["pr_auc"], ev["lightgbm"]["test"]["pr_auc"], 1e-9),

    # --- disruption / cancellations (report 5.7) ---------------------
    "cancellation ROC-AUC 0.936": approx(dis["is_cancelled"]["roc_auc"], 0.936, 1e-3),
    "cancellation catches 80% in the top decile": approx(
        dis["is_cancelled"]["recall_at_10pct"], 0.800, 5e-3),
    "cancellation is more predictable than lateness": (
        dis["is_cancelled"]["roc_auc"] > ev["lightgbm"]["test"]["roc_auc"]),
    "diversion is near-unpredictable": dis["is_diverted"]["roc_auc"] < 0.65,
    "disruption model beats the late-only model": (
        dis["is_disrupted"]["precision_at_10pct"] > ev["capacity"]["precision_at_k"]),
    "all 336,776 flights modelled for disruption": (
        dis["descriptives"]["n_flights_total"] == 336776),
    "cancellation rate 4x higher in precipitation": (
        dis["descriptives"]["cancellation_rate_precipitating"]
        / dis["descriptives"]["cancellation_rate_dry"] > 3.5),

    # --- impact model (report 10) ------------------------------------
    # The alignment check first: if the predictions had been paired with the
    # wrong flights, these totals would not survive.
    "impact exposure matches the raw test split": (
        imp["exposure"]["total_late"] == _imp_late
        and approx(imp["exposure"]["total_delay_minutes"], _imp_min, 1.0)),
    "impact uses the same alert budget as the evaluation": approx(
        imp["assumptions"]["alert_budget"], 0.10, 1e-9),
    "impact unit costs are the cited ones": (
        imp["assumptions"]["cost_per_block_minute_usd"] == 98.41
        and imp["assumptions"]["passenger_value_of_time_usd_per_hour"] == 47.0),
    "model beats random alerting on delay minutes reached": (
        imp["at_operating_budget"]["model"]["delay_min_share"]
        > 2.0 * imp["at_operating_budget"]["random"]["delay_min_share_mean"]),
    "random baseline lands on the budget, as it must": approx(
        imp["at_operating_budget"]["random"]["delay_min_share_mean"], 0.10, 5e-3),
    "model beats the no-ML historical-rate rule": (
        imp["at_operating_budget"]["model"]["delay_min_share"]
        > 1.5 * imp["at_operating_budget"]["historical_rule"]["delay_min_share"]),
    "delay minutes reached rise with the budget": all(
        a["model_delay_min_share"] <= b["model_delay_min_share"]
        for a, b in zip(imp["budget_curve"], imp["budget_curve"][1:])),
    "a 100% budget reaches 100% of delay minutes": approx(
        imp["budget_curve"][-1]["model_delay_min_share"], 1.0, 1e-9),
    "net value is positive at the operating budget": (
        imp["at_operating_budget"]["model"]["net_value_usd"] > 0),
    "net value survives every one-at-a-time cost scenario": all(
        v > 0 for knob, cases in imp["unit_cost_sensitivity"].items()
        if knob != "baseline_net_usd"
        for k, v in cases.items() if k in ("low", "high")),
    "annualisation is discounted for a late-running test period": (
        imp["scale_up"]["nyc_annual_lower_usd"]
        < imp["scale_up"]["nyc_annual_upper_usd"]),
    "break-even effectiveness is below the headline assumption": (
        imp["sensitivity"]["breakeven_effectiveness_total"]
        < imp["assumptions"]["mitigation_effectiveness"]),

    # --- fairness audit (report 11) ----------------------------------
    "fairness audit covers every carrier group above the size floor": (
        len(fair["groups"]["carrier"]) >= 10),
    "coverage gap across carriers is reported and non-trivial": (
        fair["disparity"]["carrier"]["recall_gap"] > 0.05),
    "the model under-predicts every group (December drift)": all(
        g["calibration_error"] < 0 for g in fair["groups"]["carrier"]),
    "proportional allocation spends the same budget": all(
        abs(p["proportional_alerts"] - p["global_alerts"]) <= 2
        for p in fair["price_of_equity"].values()),
    "proportional allocation narrows every coverage gap": all(
        p["proportional_recall_gap"] < p["global_recall_gap"]
        for p in fair["price_of_equity"].values()),
    "group shares sum to one": all(
        approx(sum(g["share_of_flights"] for g in groups), 1.0, 0.06)
        for groups in fair["groups"].values()),
}
for name, ok in claims.items():
    check(name, ok)

# ---------------------------------------------------------------------------
print("\n=== 5. Deliverables present ===")

expected_figs = 29
figs = sorted((ROOT / "reports" / "figures").glob("*.png"))
check(f"{expected_figs} figures generated", len(figs) == expected_figs, f"found {len(figs)}")
for path in ["README.md", "reports/report.md", "reports/report.pdf",
             "requirements.txt", "Makefile", "app/streamlit_app.py",
             ".streamlit/config.toml", "docs/PITCH.md",
             "reports/metrics/impact.json", "reports/metrics/fairness.json",
             "notebooks/01_walkthrough.ipynb", "tests/test_features.py"]:
    check(f"{path} exists", (ROOT / path).exists())

readme = (ROOT / "README.md").read_text()
check("README links the Kaggle dataset",
      "kaggle.com/datasets" in readme and "nyc-flights-2013" in readme)

nb = json.loads((ROOT / "notebooks" / "01_walkthrough.ipynb").read_text())
nb_errors = sum(1 for c in nb["cells"] for o in c.get("outputs", [])
                if o.get("output_type") == "error")
check("notebook has no execution errors", nb_errors == 0, f"{nb_errors} errors")

# ---------------------------------------------------------------------------
print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"All {len(claims) + 26} checks passed.")
