"""Central configuration for the FlightRisk NYC project.

Every tunable constant lives here so that experiments are reproducible and the
report can quote exact settings.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES, METRICS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
N_JOBS = int(os.environ.get("FLIGHTRISK_N_JOBS", "4"))

# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
# The FAA / BTS definition of an on-time arrival: within 15 minutes of schedule.
DELAY_THRESHOLD_MIN = 15
TARGET = "is_delayed"
TARGET_REG = "arr_delay"

# --------------------------------------------------------------------------
# Temporal split (the dataset covers 2013-01-01 .. 2013-12-31)
# --------------------------------------------------------------------------
# Train on the first three quarters, validate on October (model selection,
# calibration and threshold choice), and hold out Nov-Dec as a true test set.
# A purely temporal split is the only honest evaluation for an operational
# forecasting system: a random split would let the model peek at flights that
# happen *after* the ones it is scored on.
TRAIN_MONTHS = list(range(1, 9))    # Jan .. Aug   (~217k flights)
VALID_MONTHS = [9, 10]              # Sep .. Oct   (~56k flights)
TEST_MONTHS = [11, 12]              # Nov .. Dec   (~54k flights)

# The monthly late rate swings from 13.2% (September) to 32.6% (December), so
# the validation and test periods sit at genuinely different base rates. That
# is deliberate: it is exactly the seasonal shift a deployed model faces, and
# it lets us measure calibration drift rather than assume it away.

# --------------------------------------------------------------------------
# Cost model used for threshold selection
# --------------------------------------------------------------------------
# Operational framing: an airline sends a proactive re-accommodation / alert
# when the model flags a flight. A false alarm wastes a small amount of agent
# time and erodes trust; a missed delay leads to misconnections, rebooking and
# compensation. We assume a missed delay costs 4x a false alarm. This ratio is
# a *policy* choice, not a modelling one, so it is surfaced here and swept in
# the evaluation.
COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 4.0

# Fraction of flights an operations desk can realistically act on per day.
# Used for the precision@k / lift analysis.
CAPACITY_FRACTION = 0.10

# --------------------------------------------------------------------------
# Feature-mode definitions
# --------------------------------------------------------------------------
# MODE_A ("day-of planning"): everything is known >= 2 hours before the
#   scheduled departure. This is the deployable model.
# MODE_B ("gate update"): adds signals that only exist once the aircraft has
#   actually pushed back (observed departure delay, actual inbound-leg delay).
#   Included to quantify how much the departure signal is worth -- and to make
#   explicit which features would be leakage in mode A.
MODE_A = "preflight"
MODE_B = "gate"
GATE_ONLY_FEATURES = ["dep_delay", "inbound_arr_delay", "inbound_delay_known"]

# Columns that describe the *outcome* and must never be used as inputs.
LEAKY_COLUMNS = [
    "arr_delay", "arr_time", "dep_time", "air_time",
    "is_delayed", "time_hour", "actual_block_time",
]

# --------------------------------------------------------------------------
# US public holidays in 2013 (used for a travel-peak indicator)
# --------------------------------------------------------------------------
US_HOLIDAYS_2013 = [
    "2013-01-01",  # New Year's Day
    "2013-01-21",  # MLK Day
    "2013-02-18",  # Presidents' Day
    "2013-05-27",  # Memorial Day
    "2013-07-04",  # Independence Day
    "2013-09-02",  # Labor Day
    "2013-10-14",  # Columbus Day
    "2013-11-11",  # Veterans Day
    "2013-11-28",  # Thanksgiving
    "2013-12-25",  # Christmas Day
]
HOLIDAY_WINDOW_DAYS = 2  # +/- this many days counts as a peak-travel period

# --------------------------------------------------------------------------
# Target-encoding smoothing
# --------------------------------------------------------------------------
# Bayesian smoothing toward the global prior. m = number of "pseudo-flights"
# of prior evidence; groups with fewer real flights get shrunk hard.
TARGET_ENCODING_SMOOTHING = 50
TARGET_ENCODING_FOLDS = 5
