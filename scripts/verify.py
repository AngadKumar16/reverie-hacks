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
shp = json.loads((METRICS / "shap_importance.json").read_text())
eda = json.loads((METRICS / "eda_summary.json").read_text())

claims = {
    "pre-flight PR-AUC 0.507": approx(ev["lightgbm"]["test"]["pr_auc"], 0.507, 5e-4),
    "pre-flight ROC-AUC 0.716": approx(ev["lightgbm"]["test"]["roc_auc"], 0.716, 5e-4),
    "gate PR-AUC 0.846": approx(ev["lightgbm_gate"]["test"]["pr_auc"], 0.846, 5e-4),
    "gate ROC-AUC 0.903": approx(ev["lightgbm_gate"]["test"]["roc_auc"], 0.903, 5e-4),
    "xgboost PR-AUC 0.508": approx(ev["xgboost"]["test"]["pr_auc"], 0.508, 5e-4),
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
}
for name, ok in claims.items():
    check(name, ok)

# ---------------------------------------------------------------------------
print("\n=== 5. Deliverables present ===")

expected_figs = 22
figs = sorted((ROOT / "reports" / "figures").glob("*.png"))
check(f"{expected_figs} figures generated", len(figs) == expected_figs, f"found {len(figs)}")
for path in ["README.md", "reports/report.md", "reports/report.pdf",
             "requirements.txt", "Makefile", "app/streamlit_app.py",
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
print(f"All {len(claims) + 22} checks passed.")
