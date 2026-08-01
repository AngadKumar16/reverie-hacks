"""Training: baselines, a randomised search over gradient boosting, and the
final fitted artefacts.

    python -m src.train                      # run every step to completion
    python -m src.train --step search        # run one step
    python -m src.train --budget 120         # stop cleanly after ~120 seconds

Every step is **resumable**. The search checkpoints after each candidate and
skips work that is already on disk, so an interrupted run can be restarted
without losing progress and without changing the result (candidates are drawn
from a seeded sampler, so draw *i* is always the same set of hyperparameters).

Search protocol
---------------
Hyperparameters are selected with a **forward-chaining time-series split inside
the training period** (Jan-Aug): fold k trains on the earliest months and scores
the block that follows, never the reverse. Random search over 40 draws is used
rather than a grid because with nine interacting hyperparameters, random
sampling covers each individual dimension far better than a grid of equal cost.

The Sep-Oct validation block is deliberately *not* used for the search. It is
reserved for early stopping of the final refit, probability calibration, and
choosing the decision threshold, so that the Nov-Dec test set is touched exactly
once, at the very end.
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
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from xgboost import XGBClassifier

from src import models as M
from src.config import METRICS, MODE_A, MODE_B, MODELS, N_JOBS, SEED
from src.features import CATEGORICAL_FEATURES
from src.pipeline import load_splits, xy

log = logging.getLogger(__name__)

SEARCH_FILE = METRICS / "lgbm_search.json"
XGB_SEARCH_FILE = METRICS / "xgb_search.json"
N_SEARCH_DRAWS = 40
# XGBoost is a cross-check on the LightGBM result, not a competitor with an
# equal budget: 8 draws is enough to confirm the two libraries land in the same
# place, and each XGBoost fit costs ~4x a LightGBM fit because the categorical
# features have to be one-hot expanded first.
N_XGB_DRAWS = 8


class Budget:
    """Cooperative time limit so long steps can stop at a safe point."""

    def __init__(self, seconds: float | None):
        self.deadline = None if seconds is None else time.time() + seconds

    def exhausted(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline


def _sorted_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("sched_dep_utc", kind="mergesort").reset_index(drop=True)


def _load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("%s was corrupt; starting over", path.name)
    return default


# ---------------------------------------------------------------------------
# Step 1: baselines
# ---------------------------------------------------------------------------


def step_baselines(Xtr, ytr, Xva, yva, feats, budget: Budget) -> bool:
    specs = [
        ("prior", lambda: M.PriorBaseline()),
        ("historical_rate", lambda: M.HistoricalRateBaseline()),
        ("logistic_regression", lambda: M.make_logistic(feats)),
        ("random_forest", lambda: M.make_random_forest(feats)),
    ]
    for name, factory in specs:
        path = MODELS / f"{name}.joblib"
        if path.exists():
            continue
        if budget.exhausted():
            return False
        t0 = time.time()
        est = factory().fit(Xtr, ytr)
        p = est.predict_proba(Xva)[:, 1]
        joblib.dump(est, path)
        log.info("%-20s valid PR-AUC %.4f  ROC-AUC %.4f  (%.0fs)", name,
                 average_precision_score(yva, p), roc_auc_score(yva, p),
                 time.time() - t0)
    return True


# ---------------------------------------------------------------------------
# Step 2: randomised search (resumable)
# ---------------------------------------------------------------------------


def step_search(X, y, budget: Budget, n_splits: int = 3) -> bool:
    candidates = list(ParameterSampler(M.LGBM_SEARCH_SPACE,
                                       n_iter=N_SEARCH_DRAWS, random_state=SEED))
    done = _load_json(SEARCH_FILE, [])
    start = len(done)
    if start >= len(candidates):
        return True

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for i in range(start, len(candidates)):
        if budget.exhausted():
            log.info("budget reached after %d/%d draws", len(done), len(candidates))
            return False
        params = candidates[i]
        t0 = time.time()
        fold_ap, fold_auc, best_iters = [], [], []
        for tr_idx, va_idx in splitter.split(X):
            model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
            model.fit(X.iloc[tr_idx], y[tr_idx],
                      eval_set=[(X.iloc[va_idx], y[va_idx])],
                      eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            p = model.predict_proba(X.iloc[va_idx])[:, 1]
            fold_ap.append(average_precision_score(y[va_idx], p))
            fold_auc.append(roc_auc_score(y[va_idx], p))
            best_iters.append(model.best_iteration_ or model.n_estimators)

        done.append({
            "draw": i,
            "params": params,
            "mean_ap": float(np.mean(fold_ap)),
            "std_ap": float(np.std(fold_ap)),
            "fold_ap": [float(v) for v in fold_ap],
            "mean_auc": float(np.mean(fold_auc)),
            "median_best_iter": int(np.median(best_iters)),
            "seconds": round(time.time() - t0, 1),
        })
        SEARCH_FILE.write_text(json.dumps(done, indent=2, default=str))
        log.info("[%2d/%d] PR-AUC %.4f (+/-%.4f)  AUC %.4f  %.0fs",
                 i + 1, len(candidates), done[-1]["mean_ap"], done[-1]["std_ap"],
                 done[-1]["mean_auc"], done[-1]["seconds"])
    return True


def best_params() -> Dict:
    done = _load_json(SEARCH_FILE, [])
    if not done:
        raise SystemExit("run the search step first")
    return max(done, key=lambda r: r["mean_ap"])["params"]


# ---------------------------------------------------------------------------
# Step 3: final fits
# ---------------------------------------------------------------------------


def _fit_lgbm(Xtr, ytr, Xva, yva, params, label) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    log.info("%s: stopped at %d trees, valid PR-AUC %.4f", label,
             model.best_iteration_,
             average_precision_score(yva, model.predict_proba(Xva)[:, 1]))
    return model


def step_final(train, valid, feats_a, feats_b, budget: Budget) -> bool:
    params = best_params()
    Xtr, ytr = xy(train, feats_a)
    Xva, yva = xy(valid, feats_a)

    if not (MODELS / "lightgbm.joblib").exists():
        if budget.exhausted():
            return False
        joblib.dump(_fit_lgbm(Xtr, ytr, Xva, yva, params, "pre-flight"),
                    MODELS / "lightgbm.joblib")

    if not (MODELS / "lightgbm_gate.joblib").exists():
        if budget.exhausted():
            return False
        Xtr_b, _ = xy(train, feats_b)
        Xva_b, _ = xy(valid, feats_b)
        joblib.dump(_fit_lgbm(Xtr_b, ytr, Xva_b, yva, params, "post-push-back"),
                    MODELS / "lightgbm_gate.joblib")

    if not (MODELS / "lightgbm_severity.joblib").exists():
        if budget.exhausted():
            return False
        # Predict *how late* as well as *whether* late. L1 loss, because the
        # delay distribution has a heavy right tail and squared error would let
        # a handful of four-hour delays drag every prediction upward.
        reg = lgb.LGBMRegressor(objective="regression_l1", n_estimators=3000,
                                bagging_freq=1, n_jobs=N_JOBS,
                                random_state=SEED, verbose=-1, **params)
        reg.fit(Xtr, train["arr_delay"].to_numpy(),
                eval_set=[(Xva, valid["arr_delay"].to_numpy())], eval_metric="l1",
                callbacks=[lgb.early_stopping(150, verbose=False)])
        log.info("severity: stopped at %d trees", reg.best_iteration_)
        joblib.dump(reg, MODELS / "lightgbm_severity.joblib")

    joblib.dump({"features_preflight": feats_a, "features_gate": feats_b,
                 "best_params": params, "seed": SEED},
                MODELS / "training_context.joblib")
    write_fingerprints()
    return True


def write_fingerprints() -> None:
    """Record a stable checksum of each booster.

    A joblib pickle of an LGBMClassifier is *not* byte-stable across runs even
    when training is fully deterministic -- the container carries incidental
    state. The booster's own model string is stable, so hashing that gives a
    fingerprint anyone can compare after a rebuild to confirm they got the
    identical model rather than merely a similar-scoring one.
    """
    import hashlib

    out = {}
    for name in ["lightgbm", "lightgbm_gate", "lightgbm_severity"]:
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        text = model.booster_.model_to_string()
        out[name] = {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "n_trees": int(model.best_iteration_ or 0),
            "n_features": int(model.n_features_),
        }
    ctx = joblib.load(MODELS / "training_context.joblib")
    out["feature_list_sha256"] = hashlib.sha256(
        "\n".join(ctx["features_preflight"]).encode()).hexdigest()
    (METRICS / "model_fingerprints.json").write_text(json.dumps(out, indent=2))
    log.info("wrote model fingerprints")


# ---------------------------------------------------------------------------
# Step 4: XGBoost cross-check
# ---------------------------------------------------------------------------


def _onehot(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, columns=[c for c in CATEGORICAL_FEATURES
                                      if c in X.columns], dummy_na=True)


def step_xgb_search(X, y, budget: Budget, n_splits: int = 3) -> bool:
    Xn = _onehot(X)
    candidates = list(ParameterSampler(M.XGB_SEARCH_SPACE, n_iter=N_XGB_DRAWS,
                                       random_state=SEED))
    done = _load_json(XGB_SEARCH_FILE, [])
    if len(done) >= len(candidates):
        return True
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for i in range(len(done), len(candidates)):
        if budget.exhausted():
            return False
        params = candidates[i]
        fold_ap, best_iters = [], []
        for tr_idx, va_idx in splitter.split(Xn):
            model = XGBClassifier(n_estimators=2000, tree_method="hist",
                                  eval_metric="aucpr", early_stopping_rounds=100,
                                  n_jobs=N_JOBS, random_state=SEED, **params)
            model.fit(Xn.iloc[tr_idx], y[tr_idx],
                      eval_set=[(Xn.iloc[va_idx], y[va_idx])], verbose=False)
            fold_ap.append(average_precision_score(
                y[va_idx], model.predict_proba(Xn.iloc[va_idx])[:, 1]))
            best_iters.append(model.best_iteration or 2000)
        done.append({"draw": i, "params": params,
                     "mean_ap": float(np.mean(fold_ap)),
                     "median_best_iter": int(np.median(best_iters))})
        XGB_SEARCH_FILE.write_text(json.dumps(done, indent=2, default=str))
        log.info("[xgb %2d/%d] PR-AUC %.4f", i + 1, len(candidates),
                 done[-1]["mean_ap"])
    return True


def step_xgb_final(train, valid, feats_a, budget: Budget) -> bool:
    if (MODELS / "xgboost.joblib").exists():
        return True
    if budget.exhausted():
        return False
    done = _load_json(XGB_SEARCH_FILE, [])
    params = max(done, key=lambda r: r["mean_ap"])["params"]
    Xtr, ytr = xy(train, feats_a)
    Xva, yva = xy(valid, feats_a)
    Xtr_n, Xva_n = _onehot(Xtr), _onehot(Xva)
    Xva_n = Xva_n.reindex(columns=Xtr_n.columns, fill_value=0)
    model = XGBClassifier(n_estimators=2000, tree_method="hist",
                          eval_metric="aucpr", early_stopping_rounds=150,
                          n_jobs=N_JOBS, random_state=SEED, **params)
    model.fit(Xtr_n, ytr, eval_set=[(Xva_n, yva)], verbose=False)
    log.info("xgboost: %d trees, valid PR-AUC %.4f", model.best_iteration,
             average_precision_score(yva, model.predict_proba(Xva_n)[:, 1]))
    joblib.dump(model, MODELS / "xgboost.joblib")
    joblib.dump(list(Xtr_n.columns), MODELS / "xgboost_columns.joblib")
    return True


# ---------------------------------------------------------------------------


STEPS = ["baselines", "search", "final", "xgb-search", "xgb-final"]


def main(step: str, budget_s: float | None) -> int:
    budget = Budget(budget_s)
    train, valid, test, manifest = load_splits()
    train, valid = _sorted_by_time(train), _sorted_by_time(valid)
    feats_a: List[str] = manifest["features"][MODE_A]
    feats_b: List[str] = manifest["features"][MODE_B]
    Xtr, ytr = xy(train, feats_a)
    Xva, yva = xy(valid, feats_a)

    todo = STEPS if step == "all" else [step]
    complete = True
    for s in todo:
        if s == "baselines":
            ok = step_baselines(Xtr, ytr, Xva, yva, feats_a, budget)
        elif s == "search":
            ok = step_search(Xtr, ytr, budget)
        elif s == "final":
            ok = step_final(train, valid, feats_a, feats_b, budget)
        elif s == "xgb-search":
            ok = step_xgb_search(Xtr, ytr, budget)
        elif s == "xgb-final":
            ok = step_xgb_final(train, valid, feats_a, budget)
        else:
            raise SystemExit(f"unknown step {s}")
        if not ok:
            complete = False
            log.info("step '%s' incomplete -- rerun to continue", s)
            break
        log.info("step '%s' complete", s)
    return 0 if complete else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=STEPS + ["all"])
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds; stop cleanly at the next checkpoint")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    sys.exit(main(args.step, args.budget))
