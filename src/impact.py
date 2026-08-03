"""What the model is worth, in minutes, passengers, dollars and CO2.

Every other module in this project answers "is the ranking any good?".  This
one answers the question an operations director would actually ask: *if I give
one person a screen and a budget of ten alerts per hundred departures, what
comes back?*

The chain from a probability to a dollar has three links, and only the first is
ours:

1. **Ranking quality** -- measured, on held-out data, in `src/evaluate.py`.
   This module re-uses the calibrated test predictions written there, so the
   impact numbers cannot disagree with the evaluation numbers.
2. **Unit costs** -- external and cited (`src/config.py`): $98.41 per block
   minute from DOT Form 41 filings, $47/h for passenger time from the FAA's
   own benefit-cost guidance, 83% load factor from BTS.
3. **Mitigation effectiveness** -- how much of a warned delay a desk can
   actually recover. Nobody publishes this. It is the honest unknown, so it is
   never assumed: the headline uses a pessimistic 10%, the whole 0-40% range is
   swept, and the break-even value is reported so a judge can decide for
   themselves whether the number is reachable.

Three properties make the result defensible rather than decorative:

* **The comparison is against a baseline, not against zero.** An operations
  desk that alerts on 10% of flights at random already catches 10% of the
  delay minutes. The model's value is the *marginal* minutes it catches over
  random -- and over a no-ML historical-rate lookup table, which is the thing
  a real airline would otherwise use.
* **Every assumption is swept.** A single point estimate would be a guess with
  a decimal point on it.
* **Scaling up is bounded, not extrapolated.** The test period is November and
  December, the two worst months of 2013. Annualising it naively overstates
  the result, so both an upper and a lower bound are reported.

    python -m src.impact

Writes `reports/metrics/impact.json` and figures 27-28.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CAPACITY_FRACTION,
    CO2_KG_PER_DELAY_MINUTE,
    COST_PER_ALERT_USD,
    COST_PER_BLOCK_MINUTE_USD,
    DELAY_THRESHOLD_MIN,
    FIGURES,
    IMPACT_BUDGETS,
    IMPACT_RANDOM_REPEATS,
    LOAD_FACTOR,
    METRICS,
    MITIGATION_EFFECTIVENESS,
    MITIGATION_SWEEP,
    PASSENGER_VALUE_OF_TIME_USD_PER_HOUR,
    SEED,
)
from src.pipeline import load_splits

log = logging.getLogger(__name__)

ACCENT = "#c44e52"
BLUE = "#3b6978"
GREEN = "#117733"
GREY = "#666666"

PRED_FILE = METRICS / "test_predictions.npy"

# Total 2013 departures from the three NYC airports, for the scale-up section.
NYC_ANNUAL_DEPARTURES = 336_776


# ---------------------------------------------------------------------------
# Exposure: what is actually sitting in a set of flights
# ---------------------------------------------------------------------------

def _prepare(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Trim the test split to the columns the impact model needs.

    ``delay_min`` is the delay we could in principle recover: minutes late
    against schedule, floored at zero (an early arrival is not a negative
    cost), and only counted for flights that actually breached the 15-minute
    BTS threshold. Shaving four minutes off a flight that landed nine minutes
    late is worth nothing to a connecting passenger, so we do not bank it.
    """
    if len(p) != len(test):
        raise ValueError(
            f"predictions ({len(p)}) and test split ({len(test)}) disagree; "
            "re-run `make evaluate` so the two are built from the same split")

    out = pd.DataFrame({
        "risk": np.asarray(p, dtype=float),
        "arr_delay": test["arr_delay"].to_numpy(dtype=float),
        "is_delayed": test["is_delayed"].to_numpy(dtype=int),
        "carrier": test["carrier"].to_numpy(),
        "origin": test["origin"].to_numpy(),
        "dest": test["dest"].to_numpy(),
    })
    out["day"] = pd.to_datetime(test["sched_dep_utc"]).dt.floor("D").to_numpy()
    # Integer day labels. Every budget sweep re-sorts by day tens of times, and
    # lexsort on datetime64 is an order of magnitude slower than on int64.
    out["day_code"] = pd.factorize(out["day"])[0]

    seats = test["seats"].to_numpy(dtype=float) if "seats" in test else np.full(len(test), np.nan)
    # Regional jets and unmatched tail numbers leave gaps; fill with the median
    # rather than dropping the flight, and record how often we had to.
    median_seats = float(np.nanmedian(seats)) if np.isfinite(seats).any() else 100.0
    out["seats"] = np.where(np.isfinite(seats), seats, median_seats)
    out["seats_imputed"] = ~np.isfinite(seats)
    out["pax"] = out["seats"] * LOAD_FACTOR

    late = out["is_delayed"].to_numpy(dtype=bool)
    out["delay_min"] = np.where(late, np.clip(out["arr_delay"], 0, None), 0.0)
    out["pax_delay_min"] = out["delay_min"] * out["pax"]

    # A no-ML ordering an airline could build from a spreadsheet: rank by the
    # route's historical late rate. If the encoding is absent, fall back to the
    # carrier's. This is the baseline the model has to beat to justify itself.
    for col in ("te_route", "te_carrier_dest", "te_carrier"):
        if col in test:
            out["historical_rank"] = test[col].to_numpy(dtype=float)
            out.attrs["historical_source"] = col
            break
    else:  # pragma: no cover - every built split carries at least one
        out["historical_rank"] = 0.0
        out.attrs["historical_source"] = "none"

    return out


def _alert_mask(df: pd.DataFrame, score: np.ndarray, budget: float) -> np.ndarray:
    """Top ``budget`` fraction of each operating day, by ``score``.

    Per-day rather than pooled, because an operations desk does not get to
    save up December's alerts and spend them all on the 23rd. This makes the
    exercise harder than a global top-k: the model must rank well *within* a
    quiet Tuesday as well as during a storm.
    """
    if budget >= 1.0:
        return np.ones(len(df), dtype=bool)

    mask = np.zeros(len(df), dtype=bool)
    day_code = df["day_code"].to_numpy()
    order = np.lexsort((-score, day_code))
    days = day_code[order]
    # Boundaries of each day's block in the sorted order.
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        k = int(np.ceil((e - s) * budget))
        mask[order[s:s + k]] = True
    return mask


def _exposure(df: pd.DataFrame, mask: np.ndarray) -> Dict[str, float]:
    sub = df[mask]
    return {
        "n_alerts": int(mask.sum()),
        "alert_rate": float(mask.mean()),
        "late_caught": int(sub["is_delayed"].sum()),
        "precision": float(sub["is_delayed"].mean()) if len(sub) else 0.0,
        "recall": float(sub["is_delayed"].sum() / max(df["is_delayed"].sum(), 1)),
        "delay_min_caught": float(sub["delay_min"].sum()),
        "delay_min_share": float(sub["delay_min"].sum() / max(df["delay_min"].sum(), 1e-9)),
        "pax_delay_min_caught": float(sub["pax_delay_min"].sum()),
        "pax_delay_min_share": float(
            sub["pax_delay_min"].sum() / max(df["pax_delay_min"].sum(), 1e-9)),
        "pax_warned": float(sub.loc[sub["is_delayed"] == 1, "pax"].sum()),
    }


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

def _value(exp: Dict[str, float], effectiveness: float,
           cost_per_alert: float = COST_PER_ALERT_USD,
           cost_per_block_minute: float = COST_PER_BLOCK_MINUTE_USD,
           vot_per_hour: float = PASSENGER_VALUE_OF_TIME_USD_PER_HOUR,
           co2_per_min: float = CO2_KG_PER_DELAY_MINUTE) -> Dict[str, float]:
    """Turn caught delay minutes into recovered value at a given effectiveness.

    Airline-side and passenger-side value are kept apart because they land on
    different balance sheets: the airline pays the $98.41 block minute, the
    passenger pays the $47 hour. A carrier evaluating this system cares about
    the first; a regulator cares about the sum.
    """
    recovered_min = exp["delay_min_caught"] * effectiveness
    recovered_pax_min = exp["pax_delay_min_caught"] * effectiveness

    airline = recovered_min * cost_per_block_minute
    passenger = recovered_pax_min * vot_per_hour / 60.0
    program = exp["n_alerts"] * cost_per_alert

    return {
        "effectiveness": float(effectiveness),
        "recovered_delay_min": float(recovered_min),
        "recovered_pax_hours": float(recovered_pax_min / 60.0),
        "airline_value_usd": float(airline),
        "passenger_value_usd": float(passenger),
        "gross_value_usd": float(airline + passenger),
        "program_cost_usd": float(program),
        "net_value_usd": float(airline + passenger - program),
        "net_airline_only_usd": float(airline - program),
        "co2_avoided_kg": float(recovered_min * co2_per_min),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _random_exposure(df: pd.DataFrame, budget: float,
                     repeats: int = IMPACT_RANDOM_REPEATS) -> Dict[str, float]:
    """Alerting at random on the same budget, averaged over seeded draws.

    This is the number the model has to beat. A desk picking 10% of flights
    with a dartboard catches ~10% of the delay minutes, so any claim of value
    has to be stated net of it.
    """
    rng = np.random.default_rng(SEED)
    keys = ["delay_min_caught", "pax_delay_min_caught", "late_caught",
            "precision", "recall", "delay_min_share"]
    acc = {k: [] for k in keys}
    for _ in range(repeats):
        score = rng.random(len(df))
        exp = _exposure(df, _alert_mask(df, score, budget))
        for k in keys:
            acc[k].append(exp[k])
    out = {f"{k}_mean": float(np.mean(v)) for k, v in acc.items()}
    out.update({f"{k}_std": float(np.std(v)) for k, v in acc.items()})
    out["n_alerts"] = int(_alert_mask(df, rng.random(len(df)), budget).sum())
    return out


# ---------------------------------------------------------------------------
# Curves and sweeps
# ---------------------------------------------------------------------------

def budget_curve(df: pd.DataFrame,
                 effectiveness: float = MITIGATION_EFFECTIVENESS) -> pd.DataFrame:
    """Model / historical-rule / random, across the whole range of budgets."""
    rows: List[Dict[str, float]] = []
    for budget in IMPACT_BUDGETS:
        model = _exposure(df, _alert_mask(df, df["risk"].to_numpy(), budget))
        hist = _exposure(df, _alert_mask(df, df["historical_rank"].to_numpy(), budget))
        rand = _random_exposure(df, budget)

        v_model = _value(model, effectiveness)
        v_hist = _value(hist, effectiveness)
        v_rand = _value({"delay_min_caught": rand["delay_min_caught_mean"],
                         "pax_delay_min_caught": rand["pax_delay_min_caught_mean"],
                         "n_alerts": rand["n_alerts"]}, effectiveness)

        rows.append({
            "budget": budget,
            "model_precision": model["precision"],
            "model_recall": model["recall"],
            "model_delay_min_share": model["delay_min_share"],
            "hist_delay_min_share": hist["delay_min_share"],
            "random_delay_min_share": rand["delay_min_share_mean"],
            "model_net_usd": v_model["net_value_usd"],
            "hist_net_usd": v_hist["net_value_usd"],
            "random_net_usd": v_rand["net_value_usd"],
            "model_minus_random_usd": v_model["net_value_usd"] - v_rand["net_value_usd"],
            "model_minus_hist_usd": v_model["net_value_usd"] - v_hist["net_value_usd"],
            "model_pax_hours": v_model["recovered_pax_hours"],
            "model_co2_kg": v_model["co2_avoided_kg"],
        })
    return pd.DataFrame(rows)


def sensitivity(df: pd.DataFrame) -> Dict[str, object]:
    """Net value across the effectiveness x budget grid, plus break-evens.

    The break-even is the interesting cell: the smallest share of delay a desk
    must recover before the system pays for its own alert handling. If that
    number is small, the conclusion survives disagreement about the assumption.
    """
    grid = []
    for eff in MITIGATION_SWEEP:
        row = []
        for budget in IMPACT_BUDGETS:
            exp = _exposure(df, _alert_mask(df, df["risk"].to_numpy(), budget))
            row.append(_value(exp, eff)["net_value_usd"])
        grid.append(row)

    # Break-even effectiveness at the operating budget, solved directly rather
    # than read off the grid: value is linear in effectiveness, so
    #   eff* = program_cost / (value per unit of effectiveness).
    exp = _exposure(df, _alert_mask(df, df["risk"].to_numpy(), CAPACITY_FRACTION))
    per_unit = _value(exp, 1.0)
    breakeven_all = per_unit["program_cost_usd"] / max(per_unit["gross_value_usd"], 1e-9)
    breakeven_airline = per_unit["program_cost_usd"] / max(per_unit["airline_value_usd"], 1e-9)

    # The break-even effectiveness comes out near zero, which is itself the
    # finding: handling an alert is so much cheaper than the delay it targets
    # that the programme is not gated on cost. The binding question is whether
    # advance warning changes anything at all, not whether it is affordable.
    # The more legible form of the same statement is the ceiling on what a desk
    # could spend per alert before the arithmetic turns negative.
    at_headline = _value(exp, MITIGATION_EFFECTIVENESS)
    n_alerts = max(exp["n_alerts"], 1)

    return {
        "effectiveness_axis": MITIGATION_SWEEP,
        "budget_axis": IMPACT_BUDGETS,
        "net_value_usd_grid": [[float(v) for v in row] for row in grid],
        "breakeven_effectiveness_total": float(breakeven_all),
        "breakeven_effectiveness_airline_only": float(breakeven_airline),
        "value_per_unit_effectiveness_usd": per_unit["gross_value_usd"],
        "breakeven_cost_per_alert_usd": float(
            at_headline["gross_value_usd"] / n_alerts),
        "gross_value_per_alert_usd": float(at_headline["gross_value_usd"] / n_alerts),
        "delay_min_per_alert": float(exp["delay_min_caught"] / n_alerts),
        "note": ("break-even effectiveness is the share of a warned flight's "
                 "delay a desk must recover for the programme to cover its own "
                 "alert-handling cost"),
    }


def unit_cost_sensitivity(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Vary each external constant one at a time and re-read the net value.

    Answers the obvious challenge -- "your conclusion depends on that $98.41"
    -- with a measurement instead of a defence.
    """
    exp = _exposure(df, _alert_mask(df, df["risk"].to_numpy(), CAPACITY_FRACTION))
    base = _value(exp, MITIGATION_EFFECTIVENESS)["net_value_usd"]

    out: Dict[str, Dict[str, float]] = {"baseline_net_usd": {"value": float(base)}}
    scenarios = {
        "block_minute_cost": [("low", {"cost_per_block_minute": 70.0}),
                              ("high", {"cost_per_block_minute": 130.0})],
        "value_of_time": [("low", {"vot_per_hour": 30.0}),
                          ("high", {"vot_per_hour": 70.0})],
        "cost_per_alert": [("low", {"cost_per_alert": 2.0}),
                           ("high", {"cost_per_alert": 20.0})],
        "co2_per_minute": [("low", {"co2_per_min": 9.0}),
                           ("high", {"co2_per_min": 27.0})],
    }
    for knob, cases in scenarios.items():
        out[knob] = {}
        for label, kwargs in cases:
            v = _value(exp, MITIGATION_EFFECTIVENESS, **kwargs)
            out[knob][label] = float(v["net_value_usd"])
            out[knob][f"{label}_co2_kg"] = float(v["co2_avoided_kg"])
        out[knob]["swing_usd"] = float(abs(out[knob]["high"] - out[knob]["low"]))
    return out


def scale_up(df: pd.DataFrame, per_period_net: float,
             annual_late_rate: float) -> Dict[str, float]:
    """Extend a two-month result to a year, with the bias stated out loud.

    The test period is November and December, which run late more often than
    the year as a whole (25.0% against 23.7%), so multiplying by six inherits
    that. We report both ends: the naive annualisation as an upper bound, and
    the same figure discounted by the late-rate ratio as a lower bound. The
    truth is between them and we do not pretend to know where.

    Both bounds are per-flight quantities scaled by NYC's actual 2013
    departure count, not a projection onto some larger market. Extrapolating
    a three-airport model to the national network would be arithmetic, not
    evidence, so it is not attempted here.
    """
    days = int(pd.Series(df["day"]).nunique())
    test_late_rate = float(df["is_delayed"].mean())
    per_flight = per_period_net / max(len(df), 1)

    upper = per_flight * NYC_ANNUAL_DEPARTURES
    lower = upper * (annual_late_rate / max(test_late_rate, 1e-9))

    return {
        "test_days": days,
        "test_flights": int(len(df)),
        "test_late_rate": test_late_rate,
        "annual_late_rate_reference": float(annual_late_rate),
        "seasonality_discount": float(annual_late_rate / max(test_late_rate, 1e-9)),
        "net_usd_per_flight": float(per_flight),
        "net_usd_per_1000_flights": float(1000 * per_flight),
        "nyc_annual_upper_usd": float(upper),
        "nyc_annual_lower_usd": float(lower),
        "nyc_annual_departures": NYC_ANNUAL_DEPARTURES,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    log.info("wrote %s.png", name)


def fig_impact(curve: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    ax = axes[0]
    x = curve["budget"] * 100
    ax.plot(x, curve["model_delay_min_share"] * 100, "o-", color=ACCENT,
            lw=2.4, label="Model ranking")
    ax.plot(x, curve["hist_delay_min_share"] * 100, "s--", color=BLUE,
            lw=2, label="Historical-rate lookup")
    ax.plot(x, curve["random_delay_min_share"] * 100, ":", color=GREY,
            lw=2, label="Random alerting")
    ax.set_xlabel("Alert budget (% of each day's departures)")
    ax.set_ylabel("Share of all delay minutes inside the alerted set (%)")
    ax.set_title("What a fixed alert budget actually reaches")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, curve["model_net_usd"] / 1e6, "o-", color=ACCENT, lw=2.4,
            label="Model ranking")
    ax.plot(x, curve["hist_net_usd"] / 1e6, "s--", color=BLUE, lw=2,
            label="Historical-rate lookup")
    ax.plot(x, curve["random_net_usd"] / 1e6, ":", color=GREY, lw=2,
            label="Random alerting")
    ax.axvline(CAPACITY_FRACTION * 100, color=GREEN, lw=1.2, ls="-.",
               label=f"Operating budget ({CAPACITY_FRACTION:.0%})")
    ax.set_xlabel("Alert budget (% of each day's departures)")
    ax.set_ylabel("Net recovered value, Nov-Dec ($M)")
    ax.set_title(f"Net value at {MITIGATION_EFFECTIVENESS:.0%} mitigation effectiveness")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle("Operational impact on the held-out Nov-Dec 2013 period",
                 y=1.02, fontsize=15)
    _save(fig, "27_impact_curve")


def fig_sensitivity(sens: Dict[str, object]) -> None:
    grid = np.array(sens["net_value_usd_grid"]) / 1e6
    effs = sens["effectiveness_axis"]
    budgets = sens["budget_axis"]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn",
                   vmin=-np.abs(grid).max(), vmax=np.abs(grid).max())
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([f"{b:.0%}" for b in budgets])
    ax.set_yticks(range(len(effs)))
    ax.set_yticklabels([f"{e:.0%}" for e in effs])
    ax.set_xlabel("Alert budget (% of each day's departures)")
    ax.set_ylabel("Mitigation effectiveness")
    ax.set_title("Net recovered value ($M, Nov-Dec) across both assumptions\n"
                 "Green is value created; the boundary is where the programme "
                 "pays for itself")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                    fontsize=8.5,
                    color="black" if abs(grid[i, j]) < np.abs(grid).max() * 0.55 else "white")

    fig.colorbar(im, ax=ax, label="Net value ($M)")
    _save(fig, "28_impact_sensitivity")


# ---------------------------------------------------------------------------

def main() -> Dict[str, object]:
    train, valid, test, manifest = load_splits()
    if not PRED_FILE.exists():
        raise SystemExit(
            "reports/metrics/test_predictions.npy is missing. "
            "Run `make evaluate` first -- this module deliberately re-uses the "
            "evaluation's calibrated predictions so the two cannot disagree.")

    # Deliberately *not* re-sorted. `evaluate.py` writes test_predictions.npy in
    # the order `load_splits()` returns, so re-ordering here would silently pair
    # every flight with somebody else's probability -- the kind of bug that
    # makes results look plausible and be wrong. `scripts/verify.py` re-derives
    # the PR-AUC from this pairing to prove the alignment holds.
    p = np.load(PRED_FILE)
    df = _prepare(test, p)

    curve = budget_curve(df)
    sens = sensitivity(df)
    unit = unit_cost_sensitivity(df)

    at_budget = _exposure(df, _alert_mask(df, df["risk"].to_numpy(), CAPACITY_FRACTION))
    at_value = _value(at_budget, MITIGATION_EFFECTIVENESS)
    at_random = _random_exposure(df, CAPACITY_FRACTION)
    at_hist = _exposure(df, _alert_mask(df, df["historical_rank"].to_numpy(),
                                        CAPACITY_FRACTION))

    results: Dict[str, object] = {
        "assumptions": {
            "cost_per_block_minute_usd": COST_PER_BLOCK_MINUTE_USD,
            "passenger_value_of_time_usd_per_hour": PASSENGER_VALUE_OF_TIME_USD_PER_HOUR,
            "load_factor": LOAD_FACTOR,
            "cost_per_alert_usd": COST_PER_ALERT_USD,
            "mitigation_effectiveness": MITIGATION_EFFECTIVENESS,
            "co2_kg_per_delay_minute": CO2_KG_PER_DELAY_MINUTE,
            "alert_budget": CAPACITY_FRACTION,
            "seats_imputed_share": float(df["seats_imputed"].mean()),
            "historical_baseline_column": df.attrs.get("historical_source"),
        },
        "exposure": {
            "total_flights": int(len(df)),
            "total_late": int(df["is_delayed"].sum()),
            "total_delay_minutes": float(df["delay_min"].sum()),
            "total_passenger_delay_hours": float(df["pax_delay_min"].sum() / 60.0),
            "mean_delay_min_given_late": float(
                df.loc[df["is_delayed"] == 1, "delay_min"].mean()),
        },
        "at_operating_budget": {
            "budget": CAPACITY_FRACTION,
            "model": {**at_budget, **at_value},
            "historical_rule": at_hist,
            "random": at_random,
            "model_vs_random_delay_min": float(
                at_budget["delay_min_caught"] - at_random["delay_min_caught_mean"]),
            "model_vs_random_usd": float(
                at_value["net_value_usd"]
                - _value({"delay_min_caught": at_random["delay_min_caught_mean"],
                          "pax_delay_min_caught": at_random["pax_delay_min_caught_mean"],
                          "n_alerts": at_random["n_alerts"]},
                         MITIGATION_EFFECTIVENESS)["net_value_usd"]),
            "model_vs_historical_delay_min": float(
                at_budget["delay_min_caught"] - at_hist["delay_min_caught"]),
        },
        "budget_curve": curve.to_dict(orient="records"),
        "sensitivity": sens,
        "unit_cost_sensitivity": unit,
        # The annualisation reference is the late rate over the *whole* labelled
        # year, not the training months, so the seasonal discount is measured
        # against the right thing.
        "scale_up": scale_up(df, at_value["net_value_usd"], float(
            pd.concat([train["is_delayed"], valid["is_delayed"],
                       test["is_delayed"]]).mean())),
    }

    fig_impact(curve, df)
    fig_sensitivity(sens)

    (METRICS / "impact.json").write_text(json.dumps(results, indent=2))
    curve.to_csv(METRICS / "impact_budget_curve.csv", index=False)

    log.info("net value at %.0f%% budget, %.0f%% effectiveness: $%.2fM "
             "(vs random: $%.2fM)",
             CAPACITY_FRACTION * 100, MITIGATION_EFFECTIVENESS * 100,
             at_value["net_value_usd"] / 1e6,
             results["at_operating_budget"]["model_vs_random_usd"] / 1e6)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    out = main()
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("budget_curve", "sensitivity")}, indent=2))
