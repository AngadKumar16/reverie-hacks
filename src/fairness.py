"""Who does the alerting system serve, and who does it quietly skip?

A ranking model with a fixed alert budget is a rationing device. Somebody gets
the phone call and somebody does not, and "highest probability first" is not a
neutral rule -- it hands the budget to whichever groups happen to have the
highest base rate. That is efficient. It is not automatically fair, and on a
route network it has a specific victim: the small destination with three
flights a day, whose passengers have no later flight to be rebooked onto and
who are therefore the *most* exposed to a delay nobody saw coming.

So this module measures the distribution rather than asserting it is fine.

For carrier, destination size, aircraft size and time of day it reports, per
group: base rate, the share of the alert budget the group receives, recall
(the equal-opportunity criterion -- of the flights that really were late, what
fraction did we warn about?), false-positive rate, and calibration error.

It then quantifies the trade-off instead of just naming it. `proportional`
re-allocates the same total budget so each group gets alerts in proportion to
its share of departures, and the module reports what that costs in delay
minutes caught. If equity is cheap, that is a finding; if it is expensive,
that is a more useful finding.

    python -m src.fairness

Writes `reports/metrics/fairness.json` and figure 29.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CAPACITY_FRACTION, FIGURES, METRICS
from src.impact import PRED_FILE, _alert_mask, _prepare
from src.pipeline import load_splits

log = logging.getLogger(__name__)

ACCENT = "#c44e52"
BLUE = "#3b6978"
GREEN = "#117733"

# Okabe-Ito: distinguishable under the three common forms of colour blindness.
# Used everywhere a categorical palette is needed, here and in the app.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

MIN_GROUP_SIZE = 200  # below this a rate is noise, so the group is pooled


# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------

def _destination_size_band(df: pd.DataFrame) -> pd.Series:
    """Quartiles of destination traffic, as seen from NYC in the test period.

    A proxy for how much slack a passenger has: JFK-LAX has a flight every
    hour, JFK-BGR does not.
    """
    counts = df["dest"].map(df["dest"].value_counts())
    q = counts.rank(pct=True)
    return pd.cut(q, [0, 0.25, 0.5, 0.75, 1.0],
                  labels=["smallest 25% of destinations",
                          "2nd quartile", "3rd quartile",
                          "largest 25% of destinations"],
                  include_lowest=True).astype(str)


def _aircraft_band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["seats"], [0, 70, 130, 200, 1000],
                  labels=["regional (<=70 seats)", "small narrowbody (71-130)",
                          "large narrowbody (131-200)", "widebody (200+)"],
                  include_lowest=True).astype(str)


def _time_band(df: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    hour = pd.to_datetime(test["sched_dep_utc"]).dt.hour.to_numpy()
    return pd.Series(pd.cut(hour, [-1, 9, 14, 19, 24],
                            labels=["early (00-09 UTC)", "midday (10-14 UTC)",
                                    "afternoon (15-19 UTC)", "late (20-23 UTC)"]
                            ).astype(str), index=df.index)


def group_definitions(df: pd.DataFrame, test: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "carrier": pd.Series(df["carrier"], index=df.index),
        "destination_size": _destination_size_band(df),
        "aircraft_size": _aircraft_band(df),
        "time_of_day": _time_band(df, test),
        "origin": pd.Series(df["origin"], index=df.index),
    }


# ---------------------------------------------------------------------------
# Per-group audit
# ---------------------------------------------------------------------------

def audit(df: pd.DataFrame, groups: pd.Series, mask: np.ndarray) -> pd.DataFrame:
    """Rates per group at a fixed alert budget.

    ``recall`` is the equal-opportunity criterion and the one that matters
    operationally: among flights that genuinely ran late, what share did the
    group get warned about? A gap here means some passengers are systematically
    told less than others about the same event.
    """
    rows: List[Dict[str, object]] = []
    y = df["is_delayed"].to_numpy(dtype=bool)
    risk = df["risk"].to_numpy()

    for name, idx in df.groupby(groups.to_numpy()).groups.items():
        sel = df.index.isin(idx)
        n = int(sel.sum())
        if n < MIN_GROUP_SIZE:
            continue
        yg, mg, rg = y[sel], mask[sel], risk[sel]
        pos, neg = yg.sum(), (~yg).sum()
        rows.append({
            "group": str(name),
            "n": n,
            "share_of_flights": float(n / len(df)),
            "base_rate": float(yg.mean()),
            "alert_rate": float(mg.mean()),
            "share_of_alerts": float(mg.sum() / max(mask.sum(), 1)),
            "recall": float((mg & yg).sum() / pos) if pos else np.nan,
            "fpr": float((mg & ~yg).sum() / neg) if neg else np.nan,
            "precision": float(yg[mg].mean()) if mg.sum() else np.nan,
            "mean_predicted": float(rg.mean()),
            "calibration_error": float(rg.mean() - yg.mean()),
            "delay_min_caught_share": float(
                df.loc[sel & mask, "delay_min"].sum()
                / max(df.loc[sel, "delay_min"].sum(), 1e-9)),
        })
    return pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)


def disparity(table: pd.DataFrame) -> Dict[str, float]:
    """Spread statistics. Ratios, not differences, so they survive rescaling."""
    def _spread(col: str) -> Dict[str, float]:
        v = table[col].dropna()
        if v.empty:
            return {}
        return {
            f"{col}_min": float(v.min()),
            f"{col}_max": float(v.max()),
            f"{col}_gap": float(v.max() - v.min()),
            f"{col}_ratio": float(v.max() / v.min()) if v.min() > 0 else float("inf"),
        }

    out: Dict[str, float] = {}
    for col in ("recall", "alert_rate", "fpr", "precision"):
        out.update(_spread(col))
    out["max_abs_calibration_error"] = float(table["calibration_error"].abs().max())
    out["worst_calibrated_group"] = str(
        table.loc[table["calibration_error"].abs().idxmax(), "group"])
    out["lowest_recall_group"] = str(table.loc[table["recall"].idxmin(), "group"])
    return out


# ---------------------------------------------------------------------------
# The equity / efficiency trade-off, priced
# ---------------------------------------------------------------------------

def proportional_mask(df: pd.DataFrame, groups: pd.Series,
                      budget: float = CAPACITY_FRACTION) -> np.ndarray:
    """Spend the same budget, but split it across groups by share of flights.

    Two details make this a fair comparison rather than a rigged one:

    * It is still allocated **per day**, exactly like `_alert_mask`. Pooling
      across the whole test period instead would let this rule spend its whole
      budget on the storm days and beat the global rule for reasons that have
      nothing to do with equity.
    * Each day's quota is divided by largest remainder, so the total number of
      alerts matches the global rule to within a flight or two instead of being
      inflated by rounding every group up.

    Within a group the ranking is untouched. This is a different rationing rule
    over the same scores, not a handicap on the model.
    """
    mask = np.zeros(len(df), dtype=bool)
    risk = df["risk"].to_numpy()
    g = groups.to_numpy()
    day = df["day_code"].to_numpy()

    for d in np.unique(day):
        in_day = np.flatnonzero(day == d)
        k_day = int(np.ceil(len(in_day) * budget))
        if k_day == 0:
            continue

        names = pd.unique(g[in_day])
        sizes = np.array([(g[in_day] == n).sum() for n in names], dtype=float)
        exact = sizes * k_day / sizes.sum()
        quota = np.floor(exact).astype(int)

        # Largest remainder: hand the leftover alerts to the groups the
        # floor treated worst, so the day's total is exactly k_day.
        left = k_day - quota.sum()
        if left > 0:
            quota[np.argsort(-(exact - quota))[:left]] += 1

        for name, k in zip(names, quota):
            if k <= 0:
                continue
            sel = in_day[g[in_day] == name]
            mask[sel[np.argsort(-risk[sel])[:k]]] = True
    return mask


def price_of_equity(df: pd.DataFrame, groups: pd.Series,
                    budget: float = CAPACITY_FRACTION) -> Dict[str, float]:
    global_mask = _alert_mask(df, df["risk"].to_numpy(), budget)
    prop_mask = proportional_mask(df, groups, budget)

    g_tab = audit(df, groups, global_mask)
    p_tab = audit(df, groups, prop_mask)

    g_min = float(df.loc[global_mask, "delay_min"].sum())
    p_min = float(df.loc[prop_mask, "delay_min"].sum())

    return {
        "budget": float(budget),
        "global_alerts": int(global_mask.sum()),
        "proportional_alerts": int(prop_mask.sum()),
        "global_delay_min_caught": g_min,
        "proportional_delay_min_caught": p_min,
        "delay_min_cost_of_equity": g_min - p_min,
        "delay_min_cost_of_equity_pct": float(100 * (g_min - p_min) / max(g_min, 1e-9)),
        "global_recall_gap": float(g_tab["recall"].max() - g_tab["recall"].min()),
        "proportional_recall_gap": float(p_tab["recall"].max() - p_tab["recall"].min()),
        "global_alert_share_gap": float(
            (g_tab["share_of_alerts"] - g_tab["share_of_flights"]).abs().max()),
        "proportional_alert_share_gap": float(
            (p_tab["share_of_alerts"] - p_tab["share_of_flights"]).abs().max()),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_fairness(tables: Dict[str, pd.DataFrame], price: Dict[str, float]) -> None:
    keys = ["carrier", "destination_size", "aircraft_size"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))

    for ax, key in zip(axes, keys):
        tab = tables[key].sort_values("base_rate")
        y = np.arange(len(tab))
        ax.barh(y - 0.2, tab["base_rate"], height=0.38, color=BLUE,
                label="actually late")
        ax.barh(y + 0.2, tab["recall"], height=0.38, color=ACCENT,
                label="of those, warned")
        ax.set_yticks(y)
        ax.set_yticklabels(tab["group"], fontsize=9)
        ax.set_xlabel("rate")
        ax.set_title(key.replace("_", " "))
        ax.grid(alpha=0.3, axis="x")
        if key == keys[0]:
            ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.suptitle(
        "Who the alert budget reaches, at a "
        f"{CAPACITY_FRACTION:.0%} budget\n"
        f"Widest recall gap {price['global_recall_gap']:.1%}; equalising it "
        f"costs {price['delay_min_cost_of_equity_pct']:.1f}% of the delay "
        "minutes caught",
        y=1.06, fontsize=14)
    fig.savefig(FIGURES / "29_fairness.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    log.info("wrote 29_fairness.png")


# ---------------------------------------------------------------------------

def main() -> Dict[str, object]:
    train, valid, test, manifest = load_splits()
    if not PRED_FILE.exists():
        raise SystemExit("run `make evaluate` first (needs test_predictions.npy)")

    # Same alignment contract as src/impact.py: predictions are stored in
    # `load_splits()` order, so the frame must not be re-sorted here.
    p = np.load(PRED_FILE)
    df = _prepare(test, p)

    mask = _alert_mask(df, df["risk"].to_numpy(), CAPACITY_FRACTION)
    groups = group_definitions(df, test)

    tables = {k: audit(df, v, mask) for k, v in groups.items()}
    results: Dict[str, object] = {
        "budget": CAPACITY_FRACTION,
        "min_group_size": MIN_GROUP_SIZE,
        "groups": {k: v.to_dict(orient="records") for k, v in tables.items()},
        "disparity": {k: disparity(v) for k, v in tables.items()},
        "price_of_equity": {k: price_of_equity(df, v)
                            for k, v in groups.items()
                            if k in ("carrier", "destination_size")},
    }

    fig_fairness(tables, results["price_of_equity"]["destination_size"])
    (METRICS / "fairness.json").write_text(json.dumps(results, indent=2))
    for k, v in tables.items():
        v.to_csv(METRICS / f"fairness_{k}.csv", index=False)

    log.info("recall gap across carriers: %.3f",
             results["disparity"]["carrier"]["recall_gap"])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    out = main()
    print(json.dumps({"disparity": out["disparity"],
                      "price_of_equity": out["price_of_equity"]}, indent=2))
