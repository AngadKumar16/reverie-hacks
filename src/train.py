"""Training: baselines, a randomised search over gradient boosting, and the
final fitted artefacts.

    python -m src.train                 # full run (~10 min on 4 cores)
    python -m src.train --n-iter 8      # quick run

Search protocol
---------------
Hyperparameters are selected with a **forward-chaining time-series split inside
the training period** (Jan-Aug): fold k trains on months 1..k and scores months
k+1.., never on earlier data. Random search over 40 draws is used rather than a
grid: with nine interacting hyperparameters, random sampling covers each
individual dimension far better than a grid of the same budget.

The Sep-Oct validation block is deliberately *not* used for the search. It is
reserved for (a) early stopping of the final refit, (b) probability
calibration, and (c) choosing the decision threshold, so the Nov-Dec test set
is touched exactly once.
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
from src.features import CATEGORICAL_FEATURES
from src.config import METRICS, MODE_A, MODE_B, MODELS, N_JOBS, SEED
from src.pipeline import load_splits, xy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


def _sorted_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("sched_dep_utc", kind="mergesort").reset_index(drop=True)


def search_lightgbm(X: pd.DataFrame, y: np.ndarray, n_iter: int,
                    n_splits: int = 3) -> Dict:
    """Randomised search scored by PR-AUC on forward-chained folds."""
    sampler = ParameterSampler(M.LGBM_SEARCH_SPACE, n_iter=n_iter,
                               random_state=SEED)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for i, params in enumerate(sampler, 1):
        t0 = time.time()
        fold_ap, fold_auc, fold_best_iter = [], [], []
        for tr_idx, va_idx in splitter.split(X):
            model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=[(X.iloc[va_idx], y[va_idx])],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            p = model.predict_proba(X.iloc[va_idx])[:, 1]
            fold_ap.append(average_precision_score(y[va_idx], p))
            fold_auc.append(roc_auc_score(y[va_idx], p))
            fold_best_iter.append(model.best_iteration_ or model.n_estimators)

        results.append({
            "params": params,
            "mean_ap": float(np.mean(fold_ap)),
            "std_ap": float(np.std(fold_ap)),
            "mean_auc": float(np.mean(fold_auc)),
            "median_best_iter": int(np.median(fold_best_iter)),
            "seconds": round(time.time() - t0, 1),
        })
        log.info("[%2d/%d] PR-AUC %.4f (+/-%.4f)  AUC %.4f  %.0fs",
                 i, n_iter, results[-1]["mean_ap"], results[-1]["std_ap"],
                 results[-1]["mean_auc"], results[-1]["seconds"])

    results.sort(key=lambda r: -r["mean_ap"])
    return {"best": results[0], "all": results}


def search_xgboost(X: pd.DataFrame, y: np.ndarray, n_iter: int,
                   n_splits: int = 3) -> Dict:
    # XGBoost needs numeric input; one-hot the low-cardinality categoricals.
    Xn = pd.get_dummies(X, columns=[c for c in CATEGORICAL_FEATURES
                                    if c in X.columns], dummy_na=True)
    sampler = ParameterSampler(M.XGB_SEARCH_SPACE, n_iter=n_iter,
                               random_state=SEED)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for i, params in enumerate(sampler, 1):
        fold_ap = []
        best_iters = []
        for tr_idx, va_idx in splitter.split(Xn):
            model = XGBClassifier(
                n_estimators=2000, tree_method="hist", eval_metric="aucpr",
                early_stopping_rounds=100, n_jobs=N_JOBS,
                random_state=SEED, **params,
            )
            model.fit(Xn.iloc[tr_idx], y[tr_idx],
                      eval_set=[(Xn.iloc[va_idx], y[va_idx])], verbose=False)
            p = model.predict_proba(Xn.iloc[va_idx])[:, 1]
            fold_ap.append(average_precision_score(y[va_idx], p))
            best_iters.append(model.best_iteration or 2000)
        results.append({"params": params, "mean_ap": float(np.mean(fold_ap)),
                        "median_best_iter": int(np.median(best_iters))})
        log.info("[xgb %2d/%d] PR-AUC %.4f", i, n_iter, results[-1]["mean_ap"])
    results.sort(key=lambda r: -r["mean_ap"])
    return {"best": results[0], "all": results}


# ---------------------------------------------------------------------------


def fit_final_lgbm(Xtr, ytr, Xva, yva, params: Dict) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**M.LGBM_FIXED, **params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(0)],
    )
    log.info("final fit stopped at %d trees", model.best_iteration_)
    return model


def fit_severity_model(Xtr, ytr_min, Xva, yva_min, params: Dict) -> lgb.LGBMRegressor:
    """Predict *how late*, in minutes, alongside the yes/no answer.

    Trained with an L1 objective because the delay distribution has a heavy
    right tail: squared error would let a handful of four-hour delays dominate
    the fit and push every prediction upward.
    """
    reg_params = {k: v for k, v in params.items()}
    reg = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=3000, bagging_freq=1,
        n_jobs=N_JOBS, random_state=SEED, verbose=-1, **reg_params,
    )
    reg.fit(Xtr, ytr_min, eval_set=[(Xva, yva_min)], eval_metric="l1",
            callbacks=[lgb.early_stopping(150, verbose=False)])
    log.info("severity model stopped at %d trees", reg.best_iteration_)
    return reg


# ---------------------------------------------------------------------------


def main(n_iter: int, xgb_iter: int, skip_xgb: bool) -> None:
    train, valid, test, manifest = load_splits()
    train, valid = _sorted_by_time(train), _sorted_by_time(valid)

    feats_a: List[str] = manifest["features"][MODE_A]
    feats_b: List[str] = manifest["features"][MODE_B]

    Xtr, ytr = xy(train, feats_a)
    Xva, yva = xy(valid, feats_a)
    log.info("train %s  valid %s  |  %d pre-flight features",
             Xtr.shape, Xva.shape, len(feats_a))

    artefacts: Dict[str, object] = {}

    # ---- baselines --------------------------------------------------
    for name, est in [("prior", M.PriorBaseline()),
                      ("historical_rate", M.HistoricalRateBaseline()),
                      ("logistic_regression", M.make_logistic(feats_a)),
                      ("random_forest", M.make_random_forest(feats_a))]:
        t0 = time.time()
        est.fit(Xtr, ytr)
        p = est.predict_proba(Xva)[:, 1]
        log.info("%-20s valid PR-AUC %.4f  ROC-AUC %.4f  (%.0fs)", name,
                 average_precision_score(yva, p), roc_auc_score(yva, p),
                 time.time() - t0)
        artefacts[name] = est

    # ---- LightGBM search + final fit --------------------------------
    log.info("--- randomised search, %d draws x 3 forward-chained folds ---", n_iter)
    search = search_lightgbm(Xtr, ytr, n_iter=n_iter)
    best_params = search["best"]["params"]
    log.info("best CV PR-AUC %.4f with %s", search["best"]["mean_ap"], best_params)
    (METRICS / "lgbm_search.json").write_text(json.dumps(search, indent=2, default=str))

    lgbm = fit_final_lgbm(Xtr, ytr, Xva, yva, best_params)
    artefacts["lightgbm"] = lgbm

    # ---- gate-mode counterfactual -----------------------------------
    Xtr_b, _ = xy(train, feats_b)
    Xva_b, _ = xy(valid, feats_b)
    lgbm_gate = fit_final_lgbm(Xtr_b, ytr, Xva_b, yva, best_params)
    artefacts["lightgbm_gate"] = lgbm_gate

    # ---- severity (regression) head ---------------------------------
    reg = fit_severity_model(
        Xtr, train["arr_delay"].to_numpy(), Xva, valid["arr_delay"].to_numpy(),
        best_params,
    )
    artefacts["lightgbm_severity"] = reg

    # ---- XGBoost cross-check ----------------------------------------
    if not skip_xgb:
        log.info("--- XGBoost cross-check, %d draws ---", xgb_iter)
        xsearch = search_xgboost(Xtr, ytr, n_iter=xgb_iter)
        (METRICS / "xgb_search.json").write_text(json.dumps(xsearch, indent=2, default=str))
        Xtr_n = pd.get_dummies(Xtr, columns=[c for c in CATEGORICAL_FEATURES
                                             if c in Xtr.columns], dummy_na=True)
        Xva_n = pd.get_dummies(Xva, columns=[c for c in CATEGORICAL_FEATURES
                                             if c in Xva.columns], dummy_na=True)
        Xva_n = Xva_n.reindex(columns=Xtr_n.columns, fill_value=0)
        xgbm = XGBClassifier(n_estimators=2000, tree_method="hist",
                             eval_metric="aucpr", early_stopping_rounds=150,
                             n_jobs=N_JOBS, random_state=SEED,
                             **xsearch["best"]["params"])
        xgbm.fit(Xtr_n, ytr, eval_set=[(Xva_n, yva)], verbose=False)
        artefacts["xgboost"] = xgbm
        joblib.dump(list(Xtr_n.columns), MODELS / "xgboost_columns.joblib")

    # ---- persist -----------------------------------------------------
    for name, est in artefacts.items():
        joblib.dump(est, MODELS / f"{name}.joblib")
    joblib.dump({"features_preflight": feats_a, "features_gate": feats_b,
                 "best_params": best_params, "seed": SEED},
                MODELS / "training_context.joblib")
    log.info("saved %d artefacts to %s", len(artefacts) + 1, MODELS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--xgb-iter", type=int, default=10)
    ap.add_argument("--skip-xgb", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    main(args.n_iter, args.xgb_iter, args.skip_xgb)
