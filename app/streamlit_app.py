"""FlightRisk NYC -- interactive demo.

    streamlit run app/streamlit_app.py

Pick a real flight from the held-out Nov-Dec 2013 period, or build a
hypothetical one, and see the model's risk estimate, the expected delay in
minutes, and a per-flight SHAP explanation. A second tab replays the operations
desk: rank a whole day by risk and see what a fixed alerting budget would have
caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CAPACITY_FRACTION, MODE_A, MODELS
from src.explain import pretty
from src.pipeline import load_splits, xy

st.set_page_config(page_title="FlightRisk NYC", page_icon="🛫", layout="wide")

ACCENT = "#c44e52"
BLUE = "#3b6978"
GREEN = "#55a868"


# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models…")
def load_everything():
    train, valid, test, manifest = load_splits()
    cols = manifest["features"][MODE_A]
    clf = joblib.load(MODELS / "lightgbm.joblib")
    reg = joblib.load(MODELS / "lightgbm_severity.joblib")
    import shap
    explainer = shap.TreeExplainer(clf)
    test = test.sort_values("sched_dep_utc").reset_index(drop=True)
    X, y = xy(test, cols)
    p = clf.predict_proba(X)[:, 1]
    mins = reg.predict(X)
    return dict(test=test, cols=cols, clf=clf, reg=reg, explainer=explainer,
                X=X, y=y, p=p, mins=mins, valid=valid, train=train)


D = load_everything()
test, X, p, mins = D["test"], D["X"], D["p"], D["mins"]
THRESHOLD = 0.20  # chosen on the validation period by expected cost


def risk_band(prob: float) -> tuple[str, str]:
    if prob >= 0.45:
        return "HIGH", ACCENT
    if prob >= THRESHOLD:
        return "ELEVATED", "#dd8452"
    return "LOW", GREEN


def explain_flight(i: int, k: int = 9) -> pd.DataFrame:
    sv = D["explainer"].shap_values(X.iloc[[i]])
    if isinstance(sv, list):
        sv = sv[1]
    s = pd.Series(sv[0], index=X.columns)
    s = s.reindex(s.abs().sort_values(ascending=False).index[:k])
    return pd.DataFrame({
        "feature": [pretty(c) for c in s.index],
        "value": [X.iloc[i][c] for c in s.index],
        "effect": s.values,
    })


def contribution_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    d = df[::-1]
    ax.barh(d["feature"], d["effect"],
            color=[ACCENT if v > 0 else BLUE for v in d["effect"]])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("effect on log-odds of arriving late")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------

st.title("🛫 FlightRisk NYC")
st.caption(
    "Will this flight arrive more than 15 minutes late? Predicted at the "
    "scheduled departure time, **before the aircraft pushes back** — so the "
    "model never sees the observed departure delay. Held-out period: "
    "November–December 2013."
)

tab_flight, tab_day, tab_model = st.tabs(
    ["Single flight", "Operations desk", "Model card"])

# ---------------------------------------------------------------------------
# Tab 1
# ---------------------------------------------------------------------------
with tab_flight:
    left, right = st.columns([1, 1.35])

    with left:
        st.subheader("Pick a flight")
        mode = st.radio("Selection", ["Browse real flights", "Random high-risk",
                                      "Random low-risk"], horizontal=False)

        if mode == "Random high-risk":
            pool = np.argsort(-p)[:500]
            i = int(np.random.default_rng().choice(pool))
        elif mode == "Random low-risk":
            pool = np.argsort(p)[:500]
            i = int(np.random.default_rng().choice(pool))
        else:
            day = st.date_input(
                "Date", value=pd.Timestamp("2013-12-22").date(),
                min_value=test["flight_date"].min().date(),
                max_value=test["flight_date"].max().date())
            sub = test[test["flight_date"].dt.date == day]
            if sub.empty:
                st.warning("No flights on that date.")
                st.stop()
            carriers = sorted(sub["carrier"].unique())
            car = st.selectbox("Carrier", carriers)
            sub = sub[sub["carrier"] == car]
            labels = [
                f"{r.carrier}{int(r.flight)}  {r.origin}→{r.dest}  "
                f"{int(r.sched_dep_hour):02d}:{int(r.sched_dep_minute):02d}"
                for r in sub.itertuples()]
            pick = st.selectbox("Flight", labels)
            i = int(sub.index[labels.index(pick)])

        row = test.iloc[i]
        prob, exp_min = float(p[i]), float(mins[i])
        band, colour = risk_band(prob)

        st.markdown(f"### {row['carrier']}{int(row['flight'])} · "
                    f"{row['origin']} → {row['dest']}")
        st.markdown(
            f"{row['flight_date'].date()} · scheduled "
            f"{int(row['sched_dep_hour']):02d}:{int(row['sched_dep_minute']):02d} · "
            f"{int(row['distance'])} mi · {int(row['sched_block_min'])} min block")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk of arriving >15 min late", f"{prob:.0%}")
        c2.metric("Expected arrival delay", f"{exp_min:+.0f} min")
        c3.markdown(
            f"<div style='padding-top:12px'><span style='background:{colour};"
            f"color:white;padding:7px 16px;border-radius:6px;font-weight:600'>"
            f"{band}</span></div>", unsafe_allow_html=True)

        actual = float(row["arr_delay"])
        st.info(f"**Ground truth:** arrived {actual:+.0f} min "
                f"({'LATE' if actual > 15 else 'on time'}). "
                f"The model did not see this.")

        with st.expander("Conditions at the origin"):
            st.write({
                "temperature (F)": round(float(row["wx_temp"]), 1),
                "wind (mph)": round(float(row["wx_wind_speed"]), 1),
                "max gust, prev 3 h (mph)": round(float(row["wx_wind_gust_max_3h"]), 1),
                "visibility (mi)": round(float(row["wx_visib"]), 1),
                "precipitation, prev 6 h (in)": round(float(row["wx_precip_6h"]), 3),
                "departures from this airport this hour": int(row["origin_hour_deps"]),
                "schedule slack in rotation (min)":
                    None if pd.isna(row["rotation_slack_min"])
                    else int(row["rotation_slack_min"]),
            })

    with right:
        st.subheader("Why")
        contrib = explain_flight(i)
        st.pyplot(contribution_chart(contrib))
        st.caption("SHAP values: an exact additive decomposition of this single "
                   "prediction. Red pushes the flight towards *late*.")
        st.dataframe(
            contrib.assign(
                value=lambda d: d["value"].map(
                    lambda v: "—" if pd.isna(v) else
                    (f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else str(v))),
                effect=lambda d: d["effect"].map(lambda v: f"{v:+.3f}")),
            hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2
# ---------------------------------------------------------------------------
with tab_day:
    st.subheader("A day at the operations desk")
    st.caption(
        "The desk cannot act on every flight. Rank the day by predicted risk, "
        "work down the list until the alerting budget runs out, and compare "
        "what that catches against acting on nothing.")

    day = st.select_slider(
        "Date",
        options=sorted(test["flight_date"].dt.date.unique()),
        value=pd.Timestamp("2013-12-22").date())
    budget = st.slider("Share of the day's flights the desk can act on",
                       0.02, 0.40, CAPACITY_FRACTION, 0.01)

    mask = (test["flight_date"].dt.date == day).to_numpy()
    d = test[mask].copy()
    d["risk"] = p[mask]
    d["expected_delay_min"] = mins[mask]
    d = d.sort_values("risk", ascending=False)
    n_act = max(int(len(d) * budget), 1)
    caught = int(d["is_delayed"].head(n_act).sum())
    total_late = int(d["is_delayed"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flights that day", f"{len(d):,}")
    m2.metric("Actually arrived late", f"{total_late:,}",
              f"{total_late / len(d):.0%} of the day")
    m3.metric(f"Alerts issued (top {budget:.0%})", f"{n_act:,}")
    m4.metric("Late flights caught", f"{caught:,}",
              f"{caught / max(total_late, 1):.0%} of all late flights")

    precision = caught / n_act
    base = total_late / len(d)
    st.markdown(
        f"**Precision {precision:.0%} versus a {base:.0%} base rate — "
        f"a lift of {precision / max(base, 1e-9):.1f}x.** Picking flights at "
        f"random would have caught about {int(n_act * base)} of them.")

    show = d.head(n_act)[[
        "carrier", "flight", "origin", "dest", "sched_dep_hour",
        "sched_dep_minute", "risk", "expected_delay_min", "arr_delay",
        "is_delayed"]].copy()
    show["scheduled"] = (show["sched_dep_hour"].astype(int).map("{:02d}".format)
                         + ":" + show["sched_dep_minute"].astype(int).map("{:02d}".format))
    show = show.drop(columns=["sched_dep_hour", "sched_dep_minute"])
    show["risk"] = show["risk"].map("{:.0%}".format)
    show["expected_delay_min"] = show["expected_delay_min"].map("{:+.0f}".format)
    show["arr_delay"] = show["arr_delay"].map("{:+.0f}".format)
    show["outcome"] = np.where(show.pop("is_delayed") == 1, "LATE", "on time")
    st.dataframe(show, hide_index=True, width="stretch", height=430)

# ---------------------------------------------------------------------------
# Tab 3
# ---------------------------------------------------------------------------
with tab_model:
    ctx = joblib.load(MODELS / "training_context.joblib")
    st.subheader("Model card")
    st.markdown(f"""
**Task** Binary classification — will a flight departing JFK, LGA or EWR arrive
more than 15 minutes late (the FAA on-time definition)?

**Decision point** The scheduled departure time, before push-back. The observed
departure delay, actual departure time, air time and arrival time are all
excluded; a model that used them would score far higher and be useless for
planning.

**Data** NYC Flights 2013 — 336,776 flights, of which 327,346 have an arrival
delay and can be labelled. Joined to hourly airport weather, the FAA aircraft
registry, airport metadata and carrier names.

**Split** Temporal, never random. Train Jan–Aug ({len(D['train']):,} flights),
validate Sep–Oct ({len(D['valid']):,}), test Nov–Dec ({len(test):,}).

**Model** LightGBM, {len(ctx['features_preflight'])} features, hyperparameters
chosen by 40-draw random search over forward-chaining time-series folds inside
the training period.

**Held-out performance** ROC-AUC 0.716, PR-AUC 0.507 against a 25.0% base rate.
At the cost-optimal threshold of 0.20: precision 47%, recall 49%. Ranking the
riskiest 10% of flights gives 64% precision, a 2.6x lift.

**Known limitations**
- Cancelled and diverted flights are dropped, so operational disruption is
  understated.
- December is systematically under-predicted; a rolling 14-day recalibration
  cuts the Brier score from 0.180 to 0.170 and is the recommended fix.
- Weather is the observation at the scheduled hour. Forecasting further ahead
  would require replacing it with an actual forecast, which will be noisier.
- Only NYC departures are recorded, so the true inbound leg of each aircraft is
  invisible and delay propagation is only partly observable.
""")
    st.json(ctx["best_params"])
