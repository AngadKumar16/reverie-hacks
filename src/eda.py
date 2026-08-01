"""Exploratory analysis: figures and summary tables that motivate the model.

Every figure written here is referenced by the report. Run with:
    python -m src.eda
"""
from __future__ import annotations

import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DELAY_THRESHOLD_MIN, FIGURES, METRICS
from src.pipeline import load_full

log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = "mako"
ACCENT = "#c44e52"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, name: str) -> None:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote %s", path.name)


# ---------------------------------------------------------------------------

def fig_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    clipped = df["arr_delay"].clip(-60, 240)
    ax.hist(clipped, bins=90, color="#3b6978", edgecolor="none")
    ax.axvline(DELAY_THRESHOLD_MIN, color=ACCENT, lw=2.5, ls="--",
               label=f"on-time cut-off (+{DELAY_THRESHOLD_MIN} min)")
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("arrival delay (minutes, clipped to [-60, 240])")
    ax.set_ylabel("flights")
    ax.set_title("Arrival delay is sharply peaked and right-skewed")
    ax.legend(fontsize=11)

    ax = axes[1]
    counts = df["is_delayed"].value_counts().sort_index()
    bars = ax.bar(["on time", "late > 15 min"], counts.values,
                  color=["#3b6978", ACCENT])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df):.1%})",
                ha="center", va="bottom", fontsize=13)
    ax.set_ylabel("flights")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Moderate class imbalance (3:1)")
    fig.suptitle("Target definition and balance", y=1.02, fontsize=17)
    _save(fig, "01_target_distribution")


def fig_seasonality(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    m = df.groupby("month")["is_delayed"].mean()
    axes[0].plot(m.index, m.values, marker="o", color="#3b6978", lw=2.5)
    axes[0].axvspan(0.5, 8.5, alpha=.10, color="#3b6978")
    axes[0].axvspan(8.5, 10.5, alpha=.18, color="#dd8452")
    axes[0].axvspan(10.5, 12.5, alpha=.18, color=ACCENT)
    axes[0].text(4.5, m.max() * 1.02, "train", ha="center", fontsize=12)
    axes[0].text(9.5, m.max() * 1.02, "valid", ha="center", fontsize=12)
    axes[0].text(11.5, m.max() * 1.02, "test", ha="center", fontsize=12)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(MONTHS, rotation=45)
    axes[0].set_ylabel("share of flights late")
    axes[0].set_title("Month: 13% in Sep vs 33% in Dec")

    h = df.groupby("sched_dep_hour")["is_delayed"].mean()
    axes[1].plot(h.index, h.values, marker="o", color="#3b6978", lw=2.5)
    axes[1].set_xlabel("scheduled departure hour (local)")
    axes[1].set_title("Hour: delay compounds through the day")

    d = df.groupby("day_of_week")["is_delayed"].mean()
    axes[2].bar(range(7), d.values, color="#3b6978")
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(DOW)
    axes[2].set_title("Weekday: Saturday is the quiet day")

    for ax in axes:
        ax.set_ylim(0, max(m.max(), h.max(), d.max()) * 1.12)
    fig.suptitle("Systematic temporal structure in the delay rate", y=1.03, fontsize=17)
    _save(fig, "02_temporal_patterns")


def fig_carrier_origin(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    c = (df.groupby("carrier")
           .agg(rate=("is_delayed", "mean"), n=("is_delayed", "size"))
           .query("n > 1000").sort_values("rate"))
    axes[0].barh(c.index, c["rate"], color=sns.color_palette("mako", len(c)))
    axes[0].axvline(df["is_delayed"].mean(), color=ACCENT, ls="--",
                    label="overall rate")
    for i, (idx, row) in enumerate(c.iterrows()):
        axes[0].text(row["rate"] + .004, i, f"n={row['n']:,.0f}",
                     va="center", fontsize=10, color="#444")
    axes[0].set_xlabel("share of flights late")
    axes[0].set_title("Carrier (>1,000 flights)")
    axes[0].legend(fontsize=11)

    piv = df.pivot_table(index="origin", columns="sched_dep_hour",
                         values="is_delayed", aggfunc="mean")
    piv = piv.loc[:, piv.columns >= 5]
    sns.heatmap(piv, ax=axes[1], cmap="rocket_r", cbar_kws={"label": "late rate"},
                linewidths=.4)
    axes[1].set_xlabel("scheduled departure hour")
    axes[1].set_ylabel("")
    axes[1].set_title("Origin airport x hour")
    fig.suptitle("Who and where: carrier and airport effects", y=1.02, fontsize=17)
    _save(fig, "03_carrier_and_airport")


def fig_weather(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    bins = [-0.001, 0.0001, 0.01, 0.05, 0.15, 10]
    labels = ["none", "trace", "light", "moderate", "heavy"]
    df = df.assign(precip_band=pd.cut(df["wx_precip"], bins=bins, labels=labels))
    p = df.groupby("precip_band", observed=True)["is_delayed"].agg(["mean", "size"])
    axes[0].bar(p.index.astype(str), p["mean"],
                color=sns.color_palette("mako", len(p)))
    for i, (m_, n_) in enumerate(zip(p["mean"], p["size"])):
        axes[0].text(i, m_ + .006, f"{m_:.0%}\nn={n_:,}", ha="center", fontsize=10)
    axes[0].set_title("Hourly precipitation at origin")
    axes[0].set_ylabel("share of flights late")

    vb = pd.cut(df["wx_visib"], [-.01, 1, 3, 5, 8, 11],
                labels=["<1 mi", "1-3", "3-5", "5-8", "8-10"])
    v = df.groupby(vb, observed=True)["is_delayed"].agg(["mean", "size"])
    axes[1].bar(v.index.astype(str), v["mean"],
                color=sns.color_palette("mako", len(v)))
    for i, (m_, n_) in enumerate(zip(v["mean"], v["size"])):
        axes[1].text(i, m_ + .006, f"{m_:.0%}\nn={n_:,}", ha="center", fontsize=10)
    axes[1].set_title("Visibility")

    wb = pd.cut(df["wx_wind_gust_max_3h"], [-.01, 10, 20, 25, 30, 100],
                labels=["<10", "10-20", "20-25", "25-30", "30+"])
    w = df.groupby(wb, observed=True)["is_delayed"].agg(["mean", "size"])
    axes[2].bar(w.index.astype(str), w["mean"],
                color=sns.color_palette("mako", len(w)))
    for i, (m_, n_) in enumerate(zip(w["mean"], w["size"])):
        axes[2].text(i, m_ + .006, f"{m_:.0%}\nn={n_:,}", ha="center", fontsize=10)
    axes[2].set_title("Max wind gust, previous 3 h (mph)")

    for ax in axes:
        ax.set_ylim(0, .62)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Weather at the NYC origin moves the delay rate by 20+ points",
                 y=1.03, fontsize=17)
    _save(fig, "04_weather_effects")


def fig_operational(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    t = df.dropna(subset=["rotation_slack_min"])
    tb = pd.cut(t["rotation_slack_min"], [-1e9, 0, 45, 90, 180, 1e9],
                labels=["negative", "0-45 m", "45-90", "90-180", "180 m+"])
    r = t.groupby(tb, observed=True)["is_delayed"].agg(["mean", "size"])
    axes[0].bar(r.index.astype(str), r["mean"],
                color=sns.color_palette("mako", len(r)))
    for i, (m_, n_) in enumerate(zip(r["mean"], r["size"])):
        axes[0].text(i, m_ + .006, f"{m_:.0%}\nn={n_:,}", ha="center", fontsize=10)
    axes[0].set_title("Schedule slack in the rotation")
    axes[0].set_ylabel("share of flights late")

    l = df.groupby("leg_of_day")["is_delayed"].agg(["mean", "size"]).query("size > 500")
    axes[1].bar(l.index.astype(str), l["mean"],
                color=sns.color_palette("mako", len(l)))
    for i, (m_, n_) in enumerate(zip(l["mean"], l["size"])):
        axes[1].text(i, m_ + .006, f"{m_:.0%}\nn={n_:,}", ha="center", fontsize=10)
    axes[1].set_xlabel("NYC departure number for this airframe that day")
    axes[1].set_title("Delay accumulates along the rotation")

    cb = pd.qcut(df["origin_hour_deps"], 5)
    c = df.groupby(cb, observed=True)["is_delayed"].agg(["mean", "size"])
    axes[2].bar([f"Q{i+1}" for i in range(len(c))], c["mean"],
                color=sns.color_palette("mako", len(c)))
    for i, (m_, n_, iv) in enumerate(zip(c["mean"], c["size"], c.index)):
        axes[2].text(i, m_ + .006, f"{m_:.0%}\n{int(iv.left)}-{int(iv.right)}/h",
                     ha="center", fontsize=10)
    axes[2].set_title("Scheduled departures in the same hour")

    for ax in axes:
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Operational structure: turnaround, rotation position, congestion",
                 y=1.03, fontsize=17)
    _save(fig, "05_operational_drivers")


def fig_correlation(df: pd.DataFrame) -> None:
    cols = ["is_delayed", "arr_delay", "dep_delay", "distance", "sched_block_min",
            "block_slack_min", "sched_dep_hour", "origin_hour_deps",
            "origin_slot15_deps", "nyc_hour_deps", "rotation_slack_min",
            "leg_of_day", "plane_age", "seats", "wx_precip", "wx_precip_3h",
            "wx_visib", "wx_wind_speed", "wx_wind_gust_max_3h", "wx_temp",
            "wx_humid", "wx_pressure"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(13, 10.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, vmin=-.6, vmax=.6,
                annot=False, linewidths=.4, ax=ax,
                cbar_kws={"label": "Spearman rho"})
    ax.set_title("Spearman correlation among candidate predictors\n"
                 "(dep_delay shown for reference only -- it is not a pre-flight feature)",
                 fontsize=14)
    _save(fig, "06_correlation_matrix")


def fig_delay_severity(df: pd.DataFrame) -> None:
    """Not all lateness is equal -- justify a second, regression head."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    late = df[df["is_delayed"] == 1]["arr_delay"]
    bands = pd.cut(late, [15, 30, 60, 120, 10000],
                   labels=["15-30 m", "30-60 m", "1-2 h", "2 h+"])
    counts = bands.value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values,
                color=sns.color_palette("rocket_r", len(counts)))
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n{v/len(late):.0%}", ha="center",
                     va="bottom", fontsize=11)
    axes[0].set_ylim(0, counts.max() * 1.2)
    axes[0].set_title("Severity among late flights")
    axes[0].set_ylabel("flights")

    ecdf = np.sort(df["arr_delay"].values)
    axes[1].plot(ecdf, np.linspace(0, 1, len(ecdf)), color="#3b6978", lw=2.5)
    axes[1].set_xlim(-60, 300)
    axes[1].axvline(15, color=ACCENT, ls="--", label="+15 min")
    for q in (0.5, 0.9, 0.99):
        val = np.quantile(df["arr_delay"], q)
        axes[1].annotate(f"p{int(q*100)} = {val:.0f} min", (val, q),
                         textcoords="offset points", xytext=(12, -14), fontsize=11)
        axes[1].scatter([val], [q], color=ACCENT, zorder=5, s=35)
    axes[1].set_xlabel("arrival delay (minutes)")
    axes[1].set_ylabel("cumulative share of flights")
    axes[1].set_title("Empirical CDF of arrival delay")
    axes[1].legend(fontsize=11)
    fig.suptitle("A binary label hides a long tail -- the case for a severity model",
                 y=1.02, fontsize=17)
    _save(fig, "07_delay_severity")


def summary_tables(df: pd.DataFrame) -> dict:
    out = {
        "n_flights_labelled": int(len(df)),
        "date_range": [str(df["flight_date"].min().date()),
                       str(df["flight_date"].max().date())],
        "late_rate_overall": float(df["is_delayed"].mean()),
        "arr_delay_quantiles": {
            f"p{int(q*100)}": float(np.quantile(df["arr_delay"], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "late_rate_by_month": {MONTHS[int(k) - 1]: round(float(v), 4) for k, v in
                               df.groupby("month")["is_delayed"].mean().items()},
        "late_rate_by_origin": {k: round(float(v), 4) for k, v in
                                df.groupby("origin")["is_delayed"].mean().items()},
        "late_rate_by_carrier": {k: round(float(v), 4) for k, v in
                                 df.groupby("carrier")["is_delayed"].mean()
                                 .sort_values().items()},
        "late_rate_precipitating_vs_dry": {
            "dry": round(float(df.loc[df["wx_precip"] == 0, "is_delayed"].mean()), 4),
            "precipitating": round(float(df.loc[df["wx_precip"] > 0, "is_delayed"].mean()), 4),
        },
        "late_rate_low_visibility": {
            "visibility_ge_3mi": round(float(df.loc[df["wx_visib"] >= 3, "is_delayed"].mean()), 4),
            "visibility_lt_3mi": round(float(df.loc[df["wx_visib"] < 3, "is_delayed"].mean()), 4),
        },
        "missingness": {
            "aircraft_registry_unmatched": round(float(df["plane_age"].isna().mean()), 4),
            "no_same_day_rotation": round(float(df["rotation_slack_min"].isna().mean()), 4),
        },
    }
    (METRICS / "eda_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    df = load_full()
    log.info("loaded %d labelled flights", len(df))
    fig_target_distribution(df)
    fig_seasonality(df)
    fig_carrier_origin(df)
    fig_weather(df)
    fig_operational(df)
    fig_correlation(df)
    fig_delay_severity(df)
    stats = summary_tables(df)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    main()
