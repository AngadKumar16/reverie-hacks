"""Model definitions: baselines and the learned estimators.

Kept separate from the training loop so the same objects can be rebuilt by the
evaluation script and the Streamlit app without re-running a search.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import N_JOBS, SEED
from src.features import CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class PriorBaseline(BaseEstimator, ClassifierMixin):
    """Predict the training base rate for every flight.

    The floor any model must clear. Its ROC-AUC is exactly 0.5 and its
    PR-AUC equals the positive rate.
    """

    def fit(self, X, y):
        self.prior_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.prior_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class HistoricalRateBaseline(BaseEstimator, ClassifierMixin):
    """The rule an operations analyst would write without machine learning.

    "Look up how often this carrier, on this route, at this hour of day has
    been late historically, and use that number." Implemented as the mean of
    the smoothed historical-rate features already computed on the training
    period, so it is a genuine like-for-like comparison rather than a strawman.
    """

    def __init__(self, columns: List[str] | None = None):
        self.columns = columns or ["te_carrier_dest", "te_origin_sched_dep_hour"]

    def fit(self, X, y):
        self.prior_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        self.used_ = [c for c in self.columns if c in X.columns]
        return self

    def predict_proba(self, X):
        p = X[self.used_].mean(axis=1).fillna(self.prior_).to_numpy()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Linear / forest pipelines (need explicit encoding)
# ---------------------------------------------------------------------------


def _preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    cats = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    nums = [c for c in feature_cols if c not in cats]
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), nums),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                # Rare levels are pooled: a destination seen 30 times in a year
                # cannot support its own coefficient.
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=200, sparse_output=False)),
            ]), cats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_logistic(feature_cols: List[str]) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(feature_cols)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, n_jobs=N_JOBS,
                                   random_state=SEED)),
    ])


def make_random_forest(feature_cols: List[str]) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(feature_cols)),
        # Deliberately capped: an unrestricted forest on 218k rows x ~130
        # one-hot columns needs several GB and adds nothing as a baseline.
        # Depth 14 with a 50-row leaf minimum is already well past the point
        # where extra capacity changes the validation score.
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=50,
            max_features="sqrt", max_samples=0.6, n_jobs=2,
            random_state=SEED,
        )),
    ])


# ---------------------------------------------------------------------------
# Gradient boosting search spaces
# ---------------------------------------------------------------------------

LGBM_SEARCH_SPACE = {
    "num_leaves": [31, 63, 95, 127, 191, 255],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_samples": [20, 50, 100, 200, 400],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda_l1": [0.0, 0.1, 1.0, 5.0],
    "lambda_l2": [0.0, 0.5, 2.0, 10.0],
    "max_depth": [-1, 6, 8, 10, 14],
    "min_split_gain": [0.0, 0.01, 0.05],
}

# Deliberately mirrors LGBM_SEARCH_SPACE: nine dimensions, matched ranges, and
# `grow_policy="lossguide"` with `max_leaves` so XGBoost grows trees the same
# leaf-wise way LightGBM does. Without that the two libraries are being asked
# to solve slightly different problems and "XGBoost scored lower" would be a
# statement about tree-growth policy rather than about the library.
XGB_SEARCH_SPACE = {
    "max_leaves": [31, 63, 95, 127, 191, 255],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [1, 5, 20, 50, 100],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0, 5.0],
    "reg_lambda": [0.0, 0.5, 2.0, 10.0],
    "max_depth": [0, 6, 8, 10, 14],   # 0 = unlimited under lossguide
    "gamma": [0.0, 0.01, 0.05],
}

XGB_FIXED = dict(
    n_estimators=3000,
    tree_method="hist",
    grow_policy="lossguide",
    # XGBoost 2.0+ handles pandas `category` dtype natively, exactly as
    # LightGBM does. The earlier one-hot expansion was both slower and a
    # different -- worse -- representation than the one LightGBM was given.
    enable_categorical=True,
    max_cat_to_onehot=1,
    eval_metric="aucpr",
    n_jobs=N_JOBS,
    random_state=SEED,
)

LGBM_FIXED = dict(
    objective="binary",
    n_estimators=3000,
    bagging_freq=1,
    n_jobs=N_JOBS,
    random_state=SEED,
    verbose=-1,
)
