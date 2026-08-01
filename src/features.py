"""Leakage-safe feature engineering for the NYC flight-delay problem.

The single most important design constraint in this project is the **prediction
horizon**. We commit to a precise decision point and only allow information that
genuinely exists at that moment:

``preflight`` (mode A, the deployable model)
    The instant of *scheduled* departure, before the aircraft has pushed back.
    Available: the published schedule, the route, the aircraft assigned to the
    tail number, current observed weather at the NYC origin, and any statistic
    learned from **past** (training-period) flights.
    Not available: ``dep_delay``, ``dep_time``, ``air_time``, ``arr_time``.

``gate`` (mode B, diagnostic only)
    A few minutes later, once the aircraft has actually pushed back. Adds the
    observed departure delay and the arrival delay of the same airframe's
    inbound leg (guarded so it is only used when that leg had genuinely landed
    before our scheduled departure).

Comparing A and B quantifies how much of the arrival-delay signal is simply
"the plane left late" versus how much is predictable in advance.

A note on which aggregates are legitimate:
  * **Schedule-derived congestion counts** (how many flights are *scheduled* to
    leave EWR in this hour) are computed over the whole calendar. That is not
    leakage: the schedule is published months ahead and is known at prediction
    time. It contains no outcome information.
  * **Outcome-derived statistics** (a carrier's historical late rate) are fit
    on the training period only, out-of-fold inside training, and then frozen.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DELAY_THRESHOLD_MIN,
    HOLIDAY_WINDOW_DAYS,
    MODE_A,
    SEED,
    TARGET,
    TARGET_ENCODING_FOLDS,
    TARGET_ENCODING_SMOOTHING,
    TEST_MONTHS,
    TRAIN_MONTHS,
    US_HOLIDAYS_2013,
    VALID_MONTHS,
)

log = logging.getLogger(__name__)

NY_UTC_OFFSET = -5  # standard offset of the three NYC airports

CATEGORICAL_FEATURES = [
    "carrier",
    "origin",
    "dest",
    "engine",
    "manufacturer",
    "aircraft_type",
]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """Convert the dataset's HHMM integer clock format to minutes past midnight."""
    s = pd.to_numeric(s, errors="coerce")
    return (s // 100) * 60 + (s % 100)


def _clean_manufacturer(s: pd.Series) -> pd.Series:
    """The FAA registry spells the same manufacturer several ways."""
    out = s.fillna("UNKNOWN").str.upper().str.strip()
    mapping = {
        r"^AIRBUS.*": "AIRBUS",
        r"^BOEING.*": "BOEING",
        r"^EMBRAER.*": "EMBRAER",
        r"^BOMBARDIER.*|^CANADAIR.*": "BOMBARDIER",
        r"^MCDONNELL DOUGLAS.*": "MCDONNELL DOUGLAS",
        r"^CESSNA.*": "CESSNA",
        r"^GULFSTREAM.*": "GULFSTREAM",
    }
    for pattern, target in mapping.items():
        out = out.str.replace(pattern, target, regex=True)
    return out


# ---------------------------------------------------------------------------
# Stage 1: base table + target
# ---------------------------------------------------------------------------


def build_base(tables: Dict[str, pd.DataFrame],
               drop_unlabelled: bool = True) -> pd.DataFrame:
    """Assemble one row per flight with schedule fields and the targets.

    ``drop_unlabelled=True`` (the default) removes cancelled and diverted
    flights, which have no arrival delay and so cannot carry an "arrived late"
    label. That is the frame the main model trains on.

    ``drop_unlabelled=False`` keeps all 336,776 rows and is used by
    ``src/cancellations.py`` to model disruption as three outcomes rather than
    two, which is what the "cancellations are dropped" limitation asks for.
    """
    flights = tables["flights"].copy()

    # A cancelled flight never left: no departure time and no arrival time.
    # A diverted flight left but did not arrive where it was meant to, so it
    # has a departure time and no arrival delay. The two are disjoint.
    flights["is_cancelled"] = (
        flights["dep_time"].isna() & flights["arr_time"].isna()
    ).astype(int)
    flights["is_diverted"] = (
        flights["dep_time"].notna() & flights["arr_delay"].isna()
    ).astype(int)

    n_all = len(flights)
    if drop_unlabelled:
        flights = flights[flights["arr_delay"].notna()].copy()
        log.info("dropped %d/%d rows without an arrival delay (cancelled/diverted)",
                 n_all - len(flights), n_all)

    flights[TARGET] = (flights["arr_delay"] > DELAY_THRESHOLD_MIN).astype(int)
    # Any of: cancelled, diverted, or arrived more than 15 minutes late.
    flights["is_disrupted"] = (
        flights[TARGET] | flights["is_cancelled"] | flights["is_diverted"]
    ).astype(int)

    # --- exact scheduled timestamps ------------------------------------
    # `time_hour` is the scheduled departure hour expressed in UTC, produced by
    # the upstream package with proper DST handling. Adding the scheduled
    # minute gives an exact UTC departure timestamp we can sort and difference.
    flights["sched_dep_utc"] = (
        pd.to_datetime(flights["time_hour"], utc=True)
        + pd.to_timedelta(flights["minute"], unit="m")
    )
    flights["flight_date"] = pd.to_datetime(
        dict(year=flights["year"], month=flights["month"], day=flights["day"])
    )

    flights["sched_dep_min"] = _hhmm_to_minutes(flights["sched_dep_time"])
    flights["sched_arr_min"] = _hhmm_to_minutes(flights["sched_arr_time"])

    # --- scheduled block time, corrected for the timezone difference ----
    tz = tables["airports"].set_index("faa")["tz"]
    flights["dest_tz"] = flights["dest"].map(tz)
    tz_gap_min = (flights["dest_tz"].fillna(NY_UTC_OFFSET) - NY_UTC_OFFSET) * 60
    raw_block = flights["sched_arr_min"] - flights["sched_dep_min"] - tz_gap_min
    # An overnight flight has a negative raw difference; wrap into [0, 1440).
    flights["sched_block_min"] = raw_block.mod(24 * 60)

    flights["sched_arr_utc"] = flights["sched_dep_utc"] + pd.to_timedelta(
        flights["sched_block_min"], unit="m"
    )
    return flights


# ---------------------------------------------------------------------------
# Stage 2: calendar features
# ---------------------------------------------------------------------------


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["flight_date"]
    df["day_of_week"] = d.dt.dayofweek
    df["day_of_year"] = d.dt.dayofyear
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    hol = pd.to_datetime(pd.Series(US_HOLIDAYS_2013))
    peak_days = set()
    for h in hol:
        for offset in range(-HOLIDAY_WINDOW_DAYS, HOLIDAY_WINDOW_DAYS + 1):
            peak_days.add(h + pd.Timedelta(days=offset))
    df["is_holiday_period"] = d.isin(peak_days).astype(int)

    df["sched_dep_hour"] = df["hour"]
    df["sched_dep_minute"] = df["minute"]
    df["sched_arr_hour"] = (df["sched_arr_min"] // 60).astype("Int64")
    # Red-eye departures behave very differently from the morning bank.
    df["is_redeye"] = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)
    # Cyclical encodings so that 23:00 and 00:00 are neighbours.
    df["dep_hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["dep_hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# ---------------------------------------------------------------------------
# Stage 3: schedule-derived congestion (no outcome information)
# ---------------------------------------------------------------------------


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """How busy is the schedule around this flight?

    Every count below is a property of the published timetable and therefore
    fully known at prediction time.
    """
    df["dep_slot_15"] = (df["sched_dep_min"] // 15).astype("Int64")

    grp = ["flight_date", "origin", "sched_dep_hour"]
    df["origin_hour_deps"] = df.groupby(grp)["flight"].transform("size")

    grp15 = ["flight_date", "origin", "dep_slot_15"]
    df["origin_slot15_deps"] = df.groupby(grp15)["flight"].transform("size")

    df["origin_day_deps"] = df.groupby(["flight_date", "origin"])["flight"].transform("size")

    # How many NYC departures are aimed at the same destination in the same
    # hour -- a proxy for arrival-side congestion at the far end.
    df["dest_hour_arrivals"] = df.groupby(
        ["flight_date", "dest", "sched_arr_hour"]
    )["flight"].transform("size")

    df["carrier_origin_hour_deps"] = df.groupby(
        ["flight_date", "origin", "carrier", "sched_dep_hour"]
    )["flight"].transform("size")

    # Share of the airport's hourly capacity this carrier occupies.
    df["carrier_share_of_slot"] = (
        df["carrier_origin_hour_deps"] / df["origin_hour_deps"]
    )

    # System-wide load across all three NYC airports in this hour.
    df["nyc_hour_deps"] = df.groupby(["flight_date", "sched_dep_hour"])["flight"].transform("size")
    return df


# ---------------------------------------------------------------------------
# Stage 4: aircraft rotation / delay-propagation structure
# ---------------------------------------------------------------------------


def add_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features describing the airframe's schedule for the day.

    Delay propagation is the dominant mechanism in airline operations: a late
    inbound aircraft makes the next departure late. Two flavours are built:

    * ``sched_turnaround_min`` -- purely schedule-based, safe for mode A.
    * ``inbound_arr_delay``    -- the observed lateness of the inbound leg,
      only defined when that leg actually landed before our scheduled
      departure. Reserved for mode B.
    """
    df = df.sort_values(["tailnum", "sched_dep_utc"], kind="mergesort").copy()
    g = df.groupby("tailnum", sort=False)

    prev_sched_arr = g["sched_arr_utc"].shift(1)
    prev_date = g["flight_date"].shift(1)
    prev_block = g["sched_block_min"].shift(1)
    # The dataset records only flights *departing* NYC, so the airframe's
    # inbound leg back into JFK/LGA/EWR is invisible. What we can see is the
    # same aircraft making a second (or third) NYC departure later the same
    # day -- 24% of flights are such a follow-on leg.
    same_day = prev_date.eq(df["flight_date"])

    # Elapsed time between the previous leg's scheduled arrival somewhere else
    # and this departure from NYC. It necessarily contains an unobserved return
    # flight, so it is *not* a turnaround on its own.
    gap = (df["sched_dep_utc"] - prev_sched_arr).dt.total_seconds() / 60
    df["rotation_gap_min"] = gap.where(same_day & (gap >= 0))

    # Subtracting the outbound block time approximates the return flight and
    # leaves the genuine schedule slack: how much room the timetable allows for
    # two turnarounds. This is the quantity that actually predicts propagation.
    slack = gap - prev_block
    df["rotation_slack_min"] = slack.where(same_day & (gap >= 0))
    df["is_tight_turn"] = (df["rotation_slack_min"] < 60).astype(float)
    df.loc[df["rotation_slack_min"].isna(), "is_tight_turn"] = np.nan

    # Which leg of the aircraft's day is this? Later legs accumulate delay.
    df["leg_of_day"] = (
        df.groupby(["tailnum", "flight_date"], sort=False).cumcount() + 1
    )
    df["tail_legs_today"] = df.groupby(["tailnum", "flight_date"], sort=False)[
        "flight"
    ].transform("size")

    # ---- gate-mode only ------------------------------------------------
    prev_arr_delay = g["arr_delay"].shift(1)
    prev_actual_arr_utc = prev_sched_arr + pd.to_timedelta(
        prev_arr_delay.fillna(0), unit="m"
    )
    # Usable only if the inbound leg is same-day *and* had actually landed
    # before our scheduled push-back. Anything else would be information from
    # the future.
    landed_in_time = same_day & (prev_actual_arr_utc <= df["sched_dep_utc"])
    df["inbound_arr_delay"] = prev_arr_delay.where(landed_in_time)
    df["inbound_delay_known"] = landed_in_time.fillna(False).astype(int)

    return df.sort_index()


# ---------------------------------------------------------------------------
# Stage 5: weather at the origin
# ---------------------------------------------------------------------------

WEATHER_COLS = [
    "temp", "dewp", "humid", "wind_dir", "wind_speed",
    "wind_gust", "precip", "pressure", "visib",
]


def prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Clean the hourly observations and add short-window trends."""
    w = weather.copy()
    w = w.sort_values(["origin", "year", "month", "day", "hour"])

    # A handful of physically impossible wind speeds exist in the ASOS feed
    # (one 1048 mph reading). Clip rather than drop so the hour survives.
    w["wind_speed"] = w["wind_speed"].clip(upper=60)
    w["wind_gust"] = w["wind_gust"].clip(upper=90)
    w["wind_gust"] = w["wind_gust"].fillna(w["wind_speed"])

    by_origin = w.groupby("origin", sort=False)
    # Recent history is known at prediction time and captures storm build-up.
    w["precip_3h"] = by_origin["precip"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    w["precip_6h"] = by_origin["precip"].transform(
        lambda s: s.rolling(6, min_periods=1).sum()
    )
    w["visib_min_3h"] = by_origin["visib"].transform(
        lambda s: s.rolling(3, min_periods=1).min()
    )
    w["wind_gust_max_3h"] = by_origin["wind_gust"].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    w["pressure_change_3h"] = by_origin["pressure"].transform(
        lambda s: s.diff(3)
    )

    w["low_visibility"] = (w["visib"] < 3).astype(float)
    w["freezing"] = (w["temp"] < 32).astype(float)
    w["is_precipitating"] = (w["precip"] > 0).astype(float)

    keep = ["origin", "year", "month", "day", "hour"] + WEATHER_COLS + [
        "precip_3h", "precip_6h", "visib_min_3h", "wind_gust_max_3h",
        "pressure_change_3h", "low_visibility", "freezing", "is_precipitating",
    ]
    w = w[keep]

    # Daylight-saving fall-back: on 3 November 2013 the clocks went back, so
    # local hour 01:00 occurs twice at each airport. That leaves six rows whose
    # (origin, date, hour) key is not unique, and a left join on a non-unique
    # key silently duplicates flights. It happens to be harmless at a zero-hour
    # horizon because nothing departs at 01:00, but any non-zero lag maps real
    # departures onto that hour and the row count blows up. Keep the first
    # observation of the repeated hour.
    key = ["origin", "year", "month", "day", "hour"]
    dupes = int(w.duplicated(subset=key).sum())
    if dupes:
        log.info("dropped %d duplicate weather hours (DST fall-back)", dupes)
        w = w.drop_duplicates(subset=key, keep="first")

    return w.rename(columns={c: f"wx_{c}" for c in keep[5:]})


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame,
                         lag_hours: int = 0) -> pd.DataFrame:
    """Join origin weather to each flight.

    ``lag_hours`` shifts the observation used *backwards* in time: with
    ``lag_hours=3`` a flight scheduled at 18:00 is given the 15:00 observation.
    That is a persistence forecast issued three hours ahead, and it lets the
    prediction horizon be extended using only data already in hand. See
    ``src/horizon.py``. The default of 0 is the deployable configuration --
    the observation available at the scheduled departure time itself.
    """
    w = prepare_weather(weather)
    if lag_hours:
        ts = pd.to_datetime(dict(year=w["year"], month=w["month"],
                                 day=w["day"], hour=w["hour"]))
        ts = ts + pd.Timedelta(hours=lag_hours)
        w["year"], w["month"] = ts.dt.year, ts.dt.month
        w["day"], w["hour"] = ts.dt.day, ts.dt.hour
    before = len(df)
    df = df.merge(w, on=["origin", "year", "month", "day", "hour"], how="left")
    assert len(df) == before, "weather join changed the row count"

    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    missing = df[wx_cols[0]].isna().mean()
    log.info("weather unmatched for %.2f%% of flights (filled per-origin)", 100 * missing)

    # A few scheduled hours have no observation. Fill with the origin's median
    # for that month -- a defensible neutral value.
    for c in wx_cols:
        df[c] = df[c].fillna(df.groupby(["origin", "month"])[c].transform("median"))
        df[c] = df[c].fillna(df[c].median())
    return df


# ---------------------------------------------------------------------------
# Stage 6: aircraft and airport metadata
# ---------------------------------------------------------------------------


def add_metadata_features(df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    planes = tables["planes"].copy()
    planes["manufacturer"] = _clean_manufacturer(planes["manufacturer"])
    planes = planes.rename(columns={"year": "plane_year", "type": "aircraft_type"})
    planes["plane_age"] = 2013 - planes["plane_year"]
    planes.loc[planes["plane_age"] < 0, "plane_age"] = np.nan

    df = df.merge(
        planes[["tailnum", "plane_age", "engines", "seats", "engine",
                "manufacturer", "aircraft_type"]],
        on="tailnum", how="left",
    )
    log.info("aircraft registry matched for %.1f%% of flights",
             100 * df["plane_age"].notna().mean())

    airports = tables["airports"].copy()
    df = df.merge(
        airports[["faa", "lat", "lon", "alt"]].rename(
            columns={"faa": "dest", "lat": "dest_lat", "lon": "dest_lon",
                     "alt": "dest_alt"}
        ),
        on="dest", how="left",
    )

    # Route-level scale: how much traffic does this city pair carry?
    df["route"] = df["origin"] + "-" + df["dest"]
    df["route_annual_flights"] = df.groupby("route")["flight"].transform("size")
    df["dest_annual_flights"] = df.groupby("dest")["flight"].transform("size")

    # Schedule padding: how generous is this flight's block time relative to
    # the typical block on the same route? Airlines pad known-bad routes.
    route_median_block = df.groupby("route")["sched_block_min"].transform("median")
    df["block_slack_min"] = df["sched_block_min"] - route_median_block
    df["implied_speed_mph"] = df["distance"] / (df["sched_block_min"] / 60).replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Stage 7: outcome-derived encodings (train-only, out-of-fold)
# ---------------------------------------------------------------------------

TARGET_ENCODE_KEYS: List[List[str]] = [
    ["carrier"],
    ["dest"],
    ["route"],
    ["tailnum"],
    ["origin", "sched_dep_hour"],
    ["carrier", "dest"],
    ["dest", "sched_dep_hour"],
    ["carrier", "sched_dep_hour"],
]
# Note on key choice: an encoding keyed on ``month`` was tried first and
# discarded. Because the split is temporal, no test month ever appears in the
# training period, so every test row would fall back to the global prior --
# a feature that is informative in training and constant at deployment is
# actively harmful. Any key must be *recurrent* across the split boundary.


def _smoothed_rate(stats: pd.DataFrame, prior: float, m: int) -> pd.Series:
    return (stats["sum"] + prior * m) / (stats["count"] + m)


def add_target_encodings(
    train: pd.DataFrame,
    others: List[pd.DataFrame],
    m: int = TARGET_ENCODING_SMOOTHING,
    n_folds: int = TARGET_ENCODING_FOLDS,
    seed: int = SEED,
    target: str = TARGET,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    """Bayesian-smoothed historical late rates.

    Fitted **only** on the training period. Inside training we use out-of-fold
    estimates so the encoding cannot memorise a row's own label; validation and
    test receive the full-training estimate. Without the out-of-fold step, a
    high-cardinality key such as ``tailnum`` would leak the label directly and
    the model would look far better in training than it is.
    """
    from sklearn.model_selection import KFold

    prior = train[target].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    created: List[str] = []

    for keys in TARGET_ENCODE_KEYS:
        name = "te_" + "_".join(keys)
        created.append(name)

        # --- out-of-fold values for the training rows -------------------
        oof = pd.Series(np.nan, index=train.index, dtype=float)
        for fit_idx, apply_idx in kf.split(train):
            fit = train.iloc[fit_idx]
            stats = fit.groupby(keys, observed=True)[target].agg(["sum", "count"])
            rate = _smoothed_rate(stats, prior, m)
            applied = train.iloc[apply_idx]
            key_index = pd.MultiIndex.from_frame(applied[keys]) if len(keys) > 1 \
                else pd.Index(applied[keys[0]])
            oof.iloc[apply_idx] = rate.reindex(key_index).to_numpy()
        train[name] = oof.fillna(prior)

        # --- full-training values for every other split ----------------
        stats = train.groupby(keys, observed=True)[target].agg(["sum", "count"])
        rate = _smoothed_rate(stats, prior, m)
        for other in others:
            key_index = pd.MultiIndex.from_frame(other[keys]) if len(keys) > 1 \
                else pd.Index(other[keys[0]])
            other[name] = rate.reindex(key_index).to_numpy()
            other[name] = other[name].fillna(prior)

    log.info("built %d target encodings (prior late rate = %.4f)", len(created), prior)
    return train, others, created


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_feature_frame(tables: Dict[str, pd.DataFrame],
                        weather_lag_hours: int = 0,
                        drop_unlabelled: bool = True) -> pd.DataFrame:
    df = build_base(tables, drop_unlabelled=drop_unlabelled)
    df = add_calendar_features(df)
    df = add_congestion_features(df)
    df = add_rotation_features(df)
    df = add_weather_features(df, tables["weather"], lag_hours=weather_lag_hours)
    df = add_metadata_features(df, tables)
    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["month"].isin(TRAIN_MONTHS)].copy()
    va = df[df["month"].isin(VALID_MONTHS)].copy()
    te = df[df["month"].isin(TEST_MONTHS)].copy()
    log.info("split sizes -- train %d, valid %d, test %d", len(tr), len(va), len(te))
    return tr, va, te


BASE_NUMERIC = [
    "month", "day_of_week", "day_of_year", "week_of_year", "is_weekend",
    "is_holiday_period", "sched_dep_hour", "sched_dep_minute", "sched_arr_hour",
    "is_redeye", "dep_hour_sin", "dep_hour_cos",
    "distance", "sched_block_min", "block_slack_min", "implied_speed_mph",
    "origin_hour_deps", "origin_slot15_deps", "origin_day_deps",
    "dest_hour_arrivals", "carrier_origin_hour_deps", "carrier_share_of_slot",
    "nyc_hour_deps",
    "rotation_gap_min", "rotation_slack_min", "is_tight_turn",
    "leg_of_day", "tail_legs_today",
    "plane_age", "engines", "seats",
    "dest_lat", "dest_lon", "dest_alt",
    "route_annual_flights", "dest_annual_flights",
]


def feature_columns(df: pd.DataFrame, mode: str, encodings: List[str]) -> List[str]:
    wx = sorted(c for c in df.columns if c.startswith("wx_"))
    cols = BASE_NUMERIC + wx + list(encodings) + CATEGORICAL_FEATURES
    if mode != MODE_A:
        cols = cols + ["dep_delay", "inbound_arr_delay", "inbound_delay_known"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns missing from frame: {missing}")
    return cols


def as_model_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Cast categoricals so LightGBM can use its native categorical splits."""
    X = df[cols].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X
