"""Correctness and leakage tests for the feature layer.

These are the checks that decide whether the reported numbers mean anything.
Run with:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.config import (
    GATE_ONLY_FEATURES,
    LEAKY_COLUMNS,
    MODE_A,
    MODE_B,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VALID_MONTHS,
)
from src.data_loader import load_tables


@pytest.fixture(scope="module")
def built():
    tables = load_tables()
    df = F.build_feature_frame(tables)
    train, valid, test = F.temporal_split(df)
    train, (valid, test), enc = F.add_target_encodings(train, [valid, test])
    return df, train, valid, test, enc


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_preflight_features_exclude_outcome_columns(built):
    df, train, _, _, enc = built
    cols = F.feature_columns(train, MODE_A, enc)
    for bad in LEAKY_COLUMNS:
        assert bad not in cols, f"{bad} is an outcome column and must not be a feature"
    for bad in GATE_ONLY_FEATURES:
        assert bad not in cols, f"{bad} is only known after push-back"


def test_gate_mode_adds_exactly_the_post_pushback_features(built):
    df, train, _, _, enc = built
    a = set(F.feature_columns(train, MODE_A, enc))
    b = set(F.feature_columns(train, MODE_B, enc))
    assert b - a == set(GATE_ONLY_FEATURES)


def test_inbound_delay_never_comes_from_the_future(built):
    """The inbound leg may only be used if it landed before we push back."""
    df = built[0]
    used = df[df["inbound_delay_known"] == 1]
    # Reconstruct the inbound leg's actual arrival for the rows we used.
    ordered = df.sort_values(["tailnum", "sched_dep_utc"], kind="mergesort")
    g = ordered.groupby("tailnum", sort=False)
    prev_sched_arr = g["sched_arr_utc"].shift(1)
    prev_delay = g["arr_delay"].shift(1)
    actual_prev_arr = prev_sched_arr + pd.to_timedelta(prev_delay.fillna(0), unit="m")
    ok = (actual_prev_arr <= ordered["sched_dep_utc"])
    flagged = ordered["inbound_delay_known"] == 1
    assert (~ok & flagged).sum() == 0
    assert len(used) > 0


def test_target_encoding_is_out_of_fold_in_training(built):
    """A tail-number encoding fitted in-fold would be a label copy.

    We check the encoding is far from a perfect predictor: a leaked encoding
    on a 4,000-level key would push training AUC towards 1.0 on its own.
    """
    from sklearn.metrics import roc_auc_score

    _, train, _, _, _ = built
    auc = roc_auc_score(train["is_delayed"], train["te_tailnum"])
    assert 0.5 < auc < 0.62, f"te_tailnum alone reaches AUC={auc:.3f}; suspect leakage"


def test_encodings_for_heldout_splits_are_not_all_prior(built):
    """Every encoding key must recur across the temporal boundary."""
    _, train, valid, test, enc = built
    prior = train["is_delayed"].mean()
    for name in enc:
        share_default = np.isclose(test[name], prior).mean()
        assert share_default < 0.5, (
            f"{name}: {share_default:.0%} of test rows fall back to the global "
            "prior -- the key does not generalise across the split"
        )


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------

def test_splits_are_disjoint_and_ordered_in_time(built):
    _, train, valid, test = built[:4]
    assert set(train["month"]) == set(TRAIN_MONTHS)
    assert set(valid["month"]) == set(VALID_MONTHS)
    assert set(test["month"]) == set(TEST_MONTHS)
    assert train["sched_dep_utc"].max() < valid["sched_dep_utc"].min()
    assert valid["sched_dep_utc"].max() < test["sched_dep_utc"].min()


def test_no_row_loss_across_the_joins(built):
    df = built[0]
    tables = load_tables()
    labelled = tables["flights"]["arr_delay"].notna().sum()
    assert len(df) == labelled


# ---------------------------------------------------------------------------
# Feature correctness
# ---------------------------------------------------------------------------

def test_scheduled_block_time_matches_actual_air_time(built):
    """Timezone maths sanity check.

    Scheduled block time should track actual air time plus taxi. If the
    timezone correction were wrong, transcontinental routes would be off by
    exactly 180 minutes.
    """
    df = built[0]
    sub = df[df["air_time"].notna()]
    diff = sub["sched_block_min"] - sub["air_time"]
    assert diff.median() == pytest.approx(31, abs=8), (
        f"median block-minus-air-time is {diff.median():.1f} min; expected ~30 "
        "minutes of taxi time"
    )
    # Transcontinental check: JFK-LAX should be ~6h of block time, not 3h or 9h.
    lax = df[(df["origin"] == "JFK") & (df["dest"] == "LAX")]
    assert 330 < lax["sched_block_min"].median() < 400


def test_hhmm_conversion():
    s = pd.Series([517, 5, 1359, 2359, 0])
    got = F._hhmm_to_minutes(s).tolist()
    assert got == [5 * 60 + 17, 5, 13 * 60 + 59, 23 * 60 + 59, 0]


def test_congestion_counts_are_consistent(built):
    df = built[0]
    # The 15-minute count can never exceed the hourly count containing it.
    assert (df["origin_slot15_deps"] <= df["origin_hour_deps"]).all()
    assert (df["carrier_origin_hour_deps"] <= df["origin_hour_deps"]).all()
    assert (df["origin_hour_deps"] <= df["origin_day_deps"]).all()
    assert df["carrier_share_of_slot"].between(0, 1).all()


def test_rotation_gap_is_non_negative_and_same_day(built):
    df = built[0]
    t = df["rotation_gap_min"].dropna()
    assert (t >= 0).all()
    assert t.max() <= 24 * 60
    # Slack must always be gap minus the previous block time.
    sub = df[df["rotation_slack_min"].notna()]
    assert (sub["rotation_slack_min"] <= sub["rotation_gap_min"]).all()


def test_target_definition(built):
    df = built[0]
    assert (df["is_delayed"] == (df["arr_delay"] > 15).astype(int)).all()
    assert df["is_delayed"].notna().all()


def test_weather_key_is_unique(built):
    """A left join on a non-unique key silently duplicates rows.

    The DST fall-back on 3 Nov 2013 repeats local hour 01:00 at each airport.
    That is invisible at a zero-hour horizon (nothing departs at 01:00) but
    corrupts every lagged join in src/horizon.py.
    """
    tables = load_tables()
    w = F.prepare_weather(tables["weather"])
    key = ["origin", "year", "month", "day", "hour"]
    assert w.duplicated(subset=key).sum() == 0


def test_lagged_weather_join_preserves_row_count(built):
    """Every forecast horizon must join one-to-one."""
    tables = load_tables()
    base = built[0]
    for lag in (0, 6, 24):
        lagged = F.build_feature_frame(tables, weather_lag_hours=lag)
        assert len(lagged) == len(base), f"lag={lag} changed the row count"


def test_weather_join_is_complete_after_fill(built):
    df = built[0]
    wx = [c for c in df.columns if c.startswith("wx_")]
    assert len(wx) >= 15
    assert df[wx].isna().sum().sum() == 0


def test_model_frame_has_no_object_dtypes(built):
    _, train, _, _, enc = built
    X = F.as_model_frame(train, F.feature_columns(train, MODE_A, enc))
    assert list(X.select_dtypes("object").columns) == []
