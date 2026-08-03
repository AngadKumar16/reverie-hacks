"""FlightRisk NYC -- the model as something you can actually use.

    streamlit run app/streamlit_app.py

Five views, in the order a newcomer needs them:

1. **One flight** -- pick a real November or December 2013 departure, see the
   risk, and see the reasons in plain English before any of the statistics.
2. **A whole day** -- the operations-desk view: rank a day, spend a fixed alert
   budget, and watch what the budget does and does not reach.
3. **What it is worth** -- the impact model from `src/impact.py`, with the
   assumptions exposed as sliders rather than buried in a footnote.
4. **Who it reaches** -- the equity audit from `src/fairness.py`.
5. **Model card** -- scope, performance and the limitations, in one place.

Design notes, because they were decisions and not defaults:

* **Colour is never the only signal.** Every risk level carries a word and a
  shape as well as a colour, and the palette is Okabe-Ito, which survives all
  three common forms of colour blindness.
* **Every chart has a text alternative.** Not a caption -- a generated
  description of what the chart actually shows, for anyone using a screen
  reader or anyone who would simply rather read it.
* **Jargon is opt-in.** The default reading is plain English; SHAP,
  calibration and PR-AUC live behind "the technical version" expanders.
* **Two accessibility switches in the sidebar** -- high contrast and large
  text -- because "works on my laptop" is not an accessibility standard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

import streamlit as st

st.set_page_config(
    page_title="FlightRisk NYC",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"about": "FlightRisk NYC — pre-departure delay risk for JFK, "
                         "LGA and EWR. Built on NYC Flights 2013."},
)

# --- preflight -------------------------------------------------------------
# A raw ModuleNotFoundError halfway through a demo is a bad look, and the usual
# cause is mundane: streamlit is installed in one interpreter and lightgbm in
# another. Check first and say exactly what to run.
_NEEDED = {"lightgbm": "lightgbm", "shap": "shap", "joblib": "joblib",
           "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
           "matplotlib": "matplotlib", "pyarrow": "pyarrow"}
_missing = [pip for mod, pip in _NEEDED.items()
            if importlib.util.find_spec(mod) is None]

if _missing:
    st.error(f"Missing {len(_missing)} package(s) in this interpreter: "
             f"`{', '.join(_missing)}`")
    st.markdown(f"""
Streamlit is running under:

```
{sys.executable}
```

but the project's dependencies are not installed there. Install them into an
isolated environment so the two cannot drift apart:

```bash
cd {ROOT}
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make app
```

For a full diagnosis of the environment, the data and the models:

```bash
python3 scripts/doctor.py
```
""")
    st.stop()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CAPACITY_FRACTION, METRICS, MODE_A, MODELS
from src.naming import pretty
from src.pipeline import load_splits, xy

# The pipeline artefacts are gitignored, so a fresh clone has code but no data.
_ARTEFACTS = [
    (MODELS / "lightgbm.joblib", "make train"),
    (MODELS / "lightgbm_severity.joblib", "make train"),
    (ROOT / "data" / "processed" / "manifest.json", "make data"),
]
_absent = [(pth, fix) for pth, fix in _ARTEFACTS if not pth.exists()]
if _absent:
    st.error("The trained models and cached splits are not on disk yet.")
    st.markdown(
        "They are regenerated rather than committed (they weigh ~40 MB). "
        "Build them once:\n\n```bash\nmake data\nmake train\n```\n\n"
        "Missing:\n\n" + "\n".join(f"- `{p.name}` → `{fix}`" for p, fix in _absent))
    st.stop()


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Okabe-Ito. Chosen over the seaborn default because red/green is the single
# most common thing to get wrong in a risk interface: roughly 1 in 12 men and
# 1 in 200 women cannot reliably separate them.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
YELLOW = "#E69F00"
INK = "#111418"
MUTED = "#5A6472"

HIGH_CONTRAST_MAP = {BLUE: "#00325A", ORANGE: "#8A2C00",
                     GREEN: "#00503A", MUTED: "#26303C"}


# ---------------------------------------------------------------------------
# Sidebar: orientation first, then the accessibility switches
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## FlightRisk NYC")
    st.markdown(
        "Estimates the chance a flight leaving **JFK, LaGuardia or Newark** "
        "arrives more than 15 minutes late — worked out at the scheduled "
        "departure time, *before the aircraft moves*.")

    st.markdown("### Start here")
    st.markdown(
        "1. **One flight** — pick a real flight and see its risk and reasons.\n"
        "2. **A whole day** — see what a limited alert budget catches.\n"
        "3. **What it is worth** — the value of catching them.\n"
        "4. **Who it reaches** — who the budget serves, and who it skips.")

    st.markdown("### Display")
    high_contrast = st.toggle(
        "High contrast", value=False,
        help="Darkens every colour and thickens borders. Meets WCAG AA "
             "contrast for text and for graphical objects.")
    large_text = st.toggle(
        "Larger text", value=False,
        help="Increases the base font size by about 15%. Browser zoom "
             "(Ctrl or Cmd and +) works too and is respected.")
    plain_mode = st.toggle(
        "Plain English only", value=True,
        help="On: statistics stay tucked inside 'the technical version' "
             "expanders. Off: show them inline.")

    st.caption(
        "Colour is never the only signal in this app — every risk level also "
        "carries a word and a shape, and every chart has a written "
        "description.")

    with st.expander("What the words mean"):
        st.markdown(
            "**Late** — arrives more than 15 minutes behind schedule. That is "
            "the US Bureau of Transportation Statistics definition, and the "
            "one airlines are measured against.\n\n"
            "**Risk** — the model's estimated probability of that happening. "
            "A 40% risk should be wrong 6 times out of 10; that is what "
            "*calibrated* means.\n\n"
            "**Alert budget** — the share of a day's departures a desk has "
            "the staff to act on. Real desks cannot chase everything.\n\n"
            "**Held-out** — November and December 2013. The model never saw "
            "these flights while it was learning.")


if high_contrast:
    BLUE, ORANGE, GREEN, MUTED = (HIGH_CONTRAST_MAP[c]
                                  for c in (BLUE, ORANGE, GREEN, MUTED))

BASE_FONT = "1.15rem" if large_text else "1rem"
BORDER = "2px" if high_contrast else "1px"
BORDER_COLOUR = INK if high_contrast else "#D7DCE3"

st.markdown(f"""
<style>
  html, body, [class*="st-"] {{ font-size: {BASE_FONT}; }}
  .fr-hero {{
      border-left: 6px solid {BLUE};
      background: #F2F4F7;
      padding: 1.1rem 1.4rem;
      border-radius: 4px;
      margin-bottom: 1.2rem;
  }}
  .fr-hero h1 {{ font-size: 1.9rem; margin: 0 0 .35rem 0; color: {INK}; }}
  .fr-hero p  {{ margin: 0; color: {INK}; max-width: 70ch; line-height: 1.55; }}
  .fr-card {{
      border: {BORDER} solid {BORDER_COLOUR};
      border-radius: 6px;
      padding: 1rem 1.2rem;
      margin-bottom: .9rem;
      background: #FFFFFF;
  }}
  .fr-badge {{
      display: inline-block; padding: .45rem 1rem; border-radius: 4px;
      font-weight: 700; letter-spacing: .02em; color: #FFFFFF;
      font-size: 1.05rem;
  }}
  .fr-step {{ color: {MUTED}; font-size: .95rem; }}
  /* Focus ring: the Streamlit default is faint, and keyboard users need to
     be able to see where they are. */
  *:focus-visible {{ outline: 3px solid {ORANGE} !important; outline-offset: 2px; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the model…")
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

    # Severity heads (src/severity.py) and the cancellation head
    # (src/cancellations.py). Loaded opportunistically so the app still runs on
    # a partial pipeline.
    extra = {}
    for label, fname in [("p90", "lightgbm_quantile_p90.joblib"),
                         ("gt60", "lightgbm_tier_gt60.joblib"),
                         ("gt120", "lightgbm_tier_gt120.joblib")]:
        path = MODELS / fname
        if path.exists():
            m = joblib.load(path)
            extra[label] = (m.predict(X) if label == "p90"
                            else m.predict_proba(X)[:, 1])

    cancel = None
    cpath = MODELS / "lightgbm_is_cancelled.joblib"
    if cpath.exists():
        # The cancellation model was fitted with encodings refitted against the
        # cancellation target, so it carries its own feature list.
        try:
            cancel = joblib.load(cpath).predict_proba(
                X[list(joblib.load(cpath).feature_name_)])[:, 1]
        except Exception:
            cancel = None

    return dict(test=test, cols=cols, clf=clf, reg=reg, explainer=explainer,
                X=X, y=y, p=p, mins=mins, valid=valid, train=train,
                extra=extra, cancel=cancel)


@st.cache_data(show_spinner=False)
def load_metric(name: str):
    """Read a metrics JSON if the stage that writes it has been run."""
    path = METRICS / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


D = load_everything()
test, X, p, mins = D["test"], D["X"], D["p"], D["mins"]
THRESHOLD = 0.20  # chosen on the validation period by expected cost


# ---------------------------------------------------------------------------
# Small shared pieces
# ---------------------------------------------------------------------------

def risk_band(prob: float) -> tuple[str, str, str]:
    """Word, shape and colour. Never colour alone."""
    if prob >= 0.45:
        return "HIGH RISK", "▲", ORANGE
    if prob >= THRESHOLD:
        return "ELEVATED", "◆", YELLOW if not high_contrast else "#7A5200"
    return "LOW RISK", "●", GREEN


def alt_text(title: str, body: str) -> None:
    """A written stand-in for a chart.

    Screen readers cannot read a PNG, and neither can somebody reading on a
    phone in bright sun. Generated from the same numbers the chart is drawn
    from, so the two cannot drift apart.
    """
    with st.expander(f"📝 {title} — described in words"):
        st.markdown(body)


def technical(label: str = "The technical version"):
    """Jargon goes behind a door the reader chooses to open."""
    return st.expander(label, expanded=not plain_mode)


def styled_fig(figsize=(7.2, 4.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10 if not large_text else 12)
    return fig, ax


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="fr-hero">
  <h1>🛫 FlightRisk NYC</h1>
  <p>Will this flight land more than 15 minutes late? Answered at the moment
  it is scheduled to leave — <strong>before the aircraft pushes back</strong>,
  which is the last moment the answer is still worth anything. The model is
  never shown how late the flight actually departed.</p>
  <p class="fr-step" style="margin-top:.55rem">Every flight below is real, from
  November and December 2013, and none of them were used to train the model.</p>
</div>
""", unsafe_allow_html=True)

tab_flight, tab_day, tab_value, tab_fair, tab_model = st.tabs([
    "1 · One flight",
    "2 · A whole day",
    "3 · What it is worth",
    "4 · Who it reaches",
    "5 · Model card",
])


# ---------------------------------------------------------------------------
# Tab 1 -- one flight
# ---------------------------------------------------------------------------

def explain_flight(i: int, k: int = 9) -> pd.DataFrame:
    sv = D["explainer"].shap_values(X.iloc[[i]])
    if isinstance(sv, list):
        sv = sv[1]
    s = pd.Series(sv[0], index=X.columns)
    s = s.reindex(s.abs().sort_values(ascending=False).index[:k])
    return pd.DataFrame({
        "feature": [pretty(c) for c in s.index],
        "raw": list(s.index),
        "value": [X.iloc[i][c] for c in s.index],
        "effect": s.values,
    })


def contribution_chart(df: pd.DataFrame):
    fig, ax = styled_fig((7.2, 4.6))
    d = df[::-1]
    colours = [ORANGE if v > 0 else BLUE for v in d["effect"]]
    ax.barh(d["feature"], d["effect"], color=colours,
            edgecolor=INK if high_contrast else "none",
            linewidth=1.2 if high_contrast else 0)
    # Hatching so the two directions stay distinct in greyscale or in print.
    for bar, v in zip(ax.patches, d["effect"]):
        if v > 0:
            bar.set_hatch("//")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlabel("← pushes towards ON TIME     pushes towards LATE →")
    fig.tight_layout()
    return fig


def plain_reasons(contrib: pd.DataFrame) -> str:
    up = contrib[contrib["effect"] > 0].head(3)
    down = contrib[contrib["effect"] < 0].head(3)
    bits = []
    if len(up):
        bits.append("**Raising the risk:** " + ", ".join(up["feature"]) + ".")
    if len(down):
        bits.append("**Lowering it:** " + ", ".join(down["feature"]) + ".")
    return "  \n".join(bits)


with tab_flight:
    st.markdown("### Pick a flight and see what the model thinks")
    st.markdown(
        '<p class="fr-step">Choose a date and a flight on the left. The risk, '
        'the likely delay and the reasons appear on the right. What actually '
        'happened is revealed underneath — the model never saw it.</p>',
        unsafe_allow_html=True)

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        mode = st.radio(
            "How would you like to choose a flight?",
            ["Browse by date", "Show me a risky one", "Show me a safe one"],
            help="'Browse by date' lets you pick any real flight from the "
                 "held-out period. The other two pick one at random from the "
                 "500 highest or lowest risk flights.")

        if mode == "Show me a risky one":
            i = int(np.random.default_rng().choice(np.argsort(-p)[:500]))
        elif mode == "Show me a safe one":
            i = int(np.random.default_rng().choice(np.argsort(p)[:500]))
        else:
            day = st.date_input(
                "Date", value=pd.Timestamp("2013-12-22").date(),
                min_value=test["flight_date"].min().date(),
                max_value=test["flight_date"].max().date(),
                help="Any day in November or December 2013.")
            sub = test[test["flight_date"].dt.date == day]
            if sub.empty:
                st.warning("No flights on that date. Try another day.")
                st.stop()
            car = st.selectbox("Airline", sorted(sub["carrier"].unique()),
                               help="Two-letter carrier code, e.g. B6 is JetBlue.")
            sub = sub[sub["carrier"] == car]
            labels = [
                f"{r.carrier}{int(r.flight)}  {r.origin}→{r.dest}  "
                f"{int(r.sched_dep_hour):02d}:{int(r.sched_dep_minute):02d}"
                for r in sub.itertuples()]
            pick = st.selectbox("Flight", labels,
                                help="Flight number, route and scheduled "
                                     "departure time.")
            i = int(sub.index[labels.index(pick)])

        row = test.iloc[i]
        prob, exp_min = float(p[i]), float(mins[i])
        band, shape, colour = risk_band(prob)

        st.markdown(
            f'<div class="fr-card">'
            f'<div style="font-size:1.35rem;font-weight:700;color:{INK}">'
            f'{row["carrier"]}{int(row["flight"])} · {row["origin"]} → {row["dest"]}</div>'
            f'<div style="color:{MUTED};margin-top:.2rem">'
            f'{row["flight_date"].date()} · scheduled '
            f'{int(row["sched_dep_hour"]):02d}:{int(row["sched_dep_minute"]):02d} · '
            f'{int(row["distance"])} miles</div>'
            f'<div style="margin-top:.9rem">'
            f'<span class="fr-badge" style="background:{colour}">{shape} {band}</span>'
            f'</div></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("Chance of landing >15 min late", f"{prob:.0%}",
                  help="Out of 100 flights that look like this one, the model "
                       "expects about this many to arrive late.")
        c2.metric("Likely arrival, versus schedule", f"{exp_min:+.0f} min",
                  help="A point estimate. The spread matters more than the "
                       "point — see 'if it goes badly' below.")

        ex = D["extra"]
        if ex:
            st.markdown("**If it goes badly**")
            s1, s2, s3 = st.columns(3)
            if "p90" in ex:
                s1.metric("Bad-day case", f"{ex['p90'][i]:+.0f} min",
                          help="The 90th percentile: one flight in ten like "
                               "this is at least this late.")
            if "gt60" in ex:
                s2.metric("Over an hour", f"{ex['gt60'][i]:.0%}",
                          help="Roughly the point at which a connection is "
                               "lost rather than tight.")
            if "gt120" in ex:
                s3.metric("Over two hours", f"{ex['gt120'][i]:.0%}",
                          help="Overnight-hotel territory.")
            st.caption("Severe delays are easier to predict than marginal "
                       "ones — storms and broken aircraft rotations leave "
                       "traces in the data; a nine-minute delay mostly does not.")

        actual = float(row["arr_delay"])
        outcome = "arrived LATE" if actual > 15 else "arrived on time"
        (st.error if actual > 15 else st.success)(
            f"**What actually happened:** {outcome}, {actual:+.0f} minutes "
            f"against schedule. The model was not shown this.")

        with st.expander("Conditions at the airport at the time"):
            st.dataframe(pd.DataFrame({
                "measure": ["temperature (°F)", "wind (mph)",
                            "strongest gust, previous 3 h (mph)",
                            "visibility (miles)",
                            "rain or snow, previous 6 h (in)",
                            "departures from this airport this hour",
                            "spare time in the aircraft's schedule (min)"],
                "value": [
                    round(float(row["wx_temp"]), 1),
                    round(float(row["wx_wind_speed"]), 1),
                    round(float(row["wx_wind_gust_max_3h"]), 1),
                    round(float(row["wx_visib"]), 1),
                    round(float(row["wx_precip_6h"]), 3),
                    int(row["origin_hour_deps"]),
                    "unknown" if pd.isna(row["rotation_slack_min"])
                    else int(row["rotation_slack_min"]),
                ]}), hide_index=True, width="stretch")

    with right:
        st.markdown("### Why the model says that")
        contrib = explain_flight(i)
        st.markdown(plain_reasons(contrib))
        st.pyplot(contribution_chart(contrib))

        up = contrib[contrib["effect"] > 0]
        down = contrib[contrib["effect"] < 0]
        alt_text(
            "The reasons chart",
            "A horizontal bar chart of the nine things that moved this "
            "prediction most. Bars to the right (orange, hatched) push the "
            "flight towards being late; bars to the left (blue, solid) push it "
            "towards being on time.\n\n"
            + (f"Pushing later: "
               + "; ".join(f"{r.feature} ({r.effect:+.2f})"
                           for r in up.itertuples()) + ".\n\n" if len(up) else "")
            + (f"Pushing earlier: "
               + "; ".join(f"{r.feature} ({r.effect:+.2f})"
                           for r in down.itertuples()) + "." if len(down) else ""))

        with technical():
            st.markdown(
                "These are SHAP values: an exact additive decomposition of "
                "this one prediction into per-feature contributions on the "
                "log-odds scale. They sum, with the model's base value, to the "
                "logit of the probability above. Attribution is not the same "
                "as predictive value — the report's ablation study shows "
                "`day_of_year` dominates the attribution while contributing "
                "nothing to the ranking.")
            st.dataframe(
                contrib.drop(columns=["raw"]).assign(
                    value=lambda d: d["value"].map(
                        lambda v: "—" if pd.isna(v) else
                        (f"{v:.2f}" if isinstance(v, (int, float, np.floating))
                         else str(v))),
                    effect=lambda d: d["effect"].map(lambda v: f"{v:+.3f}")),
                hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 2 -- a whole day
# ---------------------------------------------------------------------------

with tab_day:
    st.markdown("### A day at the operations desk")
    st.markdown(
        '<p class="fr-step">A desk cannot chase 900 flights. It can chase a '
        'few dozen. Pick a day, set how many alerts the desk can handle, and '
        'see which flights the budget reaches — and what it misses.</p>',
        unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        day = st.select_slider(
            "Which day?",
            options=sorted(test["flight_date"].dt.date.unique()),
            value=pd.Timestamp("2013-12-22").date(),
            help="22 December 2013 is a good place to start: heavy holiday "
                 "traffic and bad weather.")
    with c2:
        budget = st.slider(
            "Alerts the desk can handle", 0.02, 0.40, CAPACITY_FRACTION, 0.01,
            format="%.0f%%",
            help="As a share of that day's departures. Ten percent is the "
                 "figure used throughout the report.")

    mask = (test["flight_date"].dt.date == day).to_numpy()
    d = test[mask].copy()
    d["risk"] = p[mask]
    d["expected_delay_min"] = mins[mask]
    d = d.sort_values("risk", ascending=False)
    n_act = max(int(len(d) * budget), 1)
    caught = int(d["is_delayed"].head(n_act).sum())
    total_late = int(d["is_delayed"].sum())
    precision = caught / n_act
    base = total_late / len(d)
    minutes_caught = float(d["arr_delay"].head(n_act).clip(lower=0).sum())
    minutes_total = float(d["arr_delay"].clip(lower=0).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flights that day", f"{len(d):,}")
    m2.metric("Arrived late", f"{total_late:,}", f"{base:.0%} of the day")
    m3.metric("Alerts issued", f"{n_act:,}", f"top {budget:.0%} by risk")
    m4.metric("Late flights warned about", f"{caught:,}",
              f"{caught / max(total_late, 1):.0%} of all late flights")

    st.markdown(
        f'<div class="fr-card">'
        f'<strong>{precision:.0%} of the alerts were right</strong>, against a '
        f'{base:.0%} chance of being right by picking at random — a lift of '
        f'{precision / max(base, 1e-9):.1f}×. Working down the list this far '
        f'reaches <strong>{minutes_caught:,.0f} of the day\'s '
        f'{minutes_total:,.0f} delay minutes</strong> '
        f'({minutes_caught / max(minutes_total, 1):.0%}).'
        f'</div>', unsafe_allow_html=True)

    alt_text("This day's result", (
        f"On {day} there were {len(d):,} departures from the three New York "
        f"airports, of which {total_late:,} ({base:.0%}) arrived more than 15 "
        f"minutes late. Alerting on the {n_act:,} riskiest flights "
        f"({budget:.0%} of the day) flags {caught:,} of those late flights, a "
        f"precision of {precision:.0%} and a lift of "
        f"{precision / max(base, 1e-9):.1f} times the base rate. Those alerts "
        f"cover {minutes_caught:,.0f} of the {minutes_total:,.0f} delay "
        f"minutes the day produced."))

    st.markdown("#### The alert list")
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
    show = show.rename(columns={
        "carrier": "airline", "flight": "no.", "origin": "from",
        "dest": "to", "risk": "risk", "expected_delay_min": "likely delay",
        "arr_delay": "actual"})
    st.dataframe(show, hide_index=True, width="stretch", height=430)
    st.caption("The 'actual' and 'outcome' columns are ground truth, shown "
               "here only so you can score the list yourself. The model saw "
               "neither.")

    with technical():
        st.markdown(
            "Ranking is by calibrated probability, and the budget is spent "
            "within the day rather than pooled across the test period. Pooling "
            "makes the numbers look better — it lets the budget be spent "
            "entirely on storm days — so `src/impact.py` uses the per-day rule "
            "throughout, and the headline precision there (47% at a 10% "
            "budget) is correspondingly lower than the 64% you get from a "
            "global top-decile.")


# ---------------------------------------------------------------------------
# Tab 3 -- what it is worth
# ---------------------------------------------------------------------------

with tab_value:
    st.markdown("### What catching them is worth")
    imp = load_metric("impact.json")

    if imp is None:
        st.info("Run `make impact` to generate `reports/metrics/impact.json`, "
                "then reload this page.")
    else:
        a = imp["assumptions"]
        e = imp["exposure"]
        m = imp["at_operating_budget"]["model"]
        su = imp["scale_up"]
        sens = imp["sensitivity"]

        st.markdown(
            '<p class="fr-step">Delay costs money on two separate balance '
            'sheets: the airline pays for the aircraft, the crew and the fuel; '
            'the passenger pays in time. Both are priced from published '
            'figures. What nobody publishes is how much of a delay advance '
            'warning actually recovers — so that is a slider, not an '
            'assumption.</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            eff = st.slider(
                "How much of a warned delay can a desk recover?",
                0.0, 0.40, float(a["mitigation_effectiveness"]), 0.01,
                format="%.0f%%",
                help="Swapping in a spare airframe, protecting a connection, "
                     "pre-positioning a crew. The report's headline uses a "
                     "deliberately pessimistic 10%.")
        with c2:
            st.metric("Delay minutes inside the alerts",
                      f"{m['delay_min_caught']:,.0f}",
                      f"{m['delay_min_share']:.0%} of all delay minutes")

        # Value is linear in effectiveness, so the slider is exact, not a refit.
        scale = eff / max(a["mitigation_effectiveness"], 1e-9)
        airline = m["airline_value_usd"] * scale
        passenger = m["passenger_value_usd"] * scale
        co2 = m["co2_avoided_kg"] * scale
        pax_hours = m["recovered_pax_hours"] * scale

        v1, v2, v3 = st.columns(3)
        v1.metric("Airline operating cost avoided", f"${airline / 1e6:,.2f}M",
                  help="At $98.41 per block minute — crew, fuel, maintenance "
                       "and ownership, from DOT Form 41 filings via Airlines "
                       "for America.")
        v2.metric("Passenger time returned", f"{pax_hours:,.0f} hours",
                  f"${passenger / 1e6:,.2f}M at the FAA's $47/hour")
        v3.metric("CO₂ not burned", f"{co2 / 1000:,.0f} tonnes",
                  help="At roughly 18 kg per delay minute. An "
                       "order-of-magnitude figure, swept in the report.")

        st.markdown(
            f'<div class="fr-card">Over the two held-out months this is '
            f'<strong>${(airline + passenger) / 1e6:,.2f}M</strong> of '
            f'recovered value against <strong>'
            f'${m["program_cost_usd"] / 1e3:,.0f}k</strong> of desk time. '
            f'Scaled to a full year of New York departures, between '
            f'<strong>${su["nyc_annual_lower_usd"] / 1e6:,.1f}M</strong> and '
            f'<strong>${su["nyc_annual_upper_usd"] / 1e6:,.1f}M</strong> — the '
            f'range is the seasonal correction, because November and December '
            f'run later than the rest of the year.</div>',
            unsafe_allow_html=True)

        st.markdown("#### It has to beat the alternatives, not beat nothing")
        curve = pd.DataFrame(imp["budget_curve"])
        fig, ax = styled_fig((8.4, 4.4))
        ax.plot(curve["budget"] * 100, curve["model_delay_min_share"] * 100,
                "o-", color=ORANGE, lw=2.6, label="This model")
        ax.plot(curve["budget"] * 100, curve["hist_delay_min_share"] * 100,
                "s--", color=BLUE, lw=2.2, label="Route's historical late rate")
        ax.plot(curve["budget"] * 100, curve["random_delay_min_share"] * 100,
                "^:", color=MUTED, lw=2.2, label="Picking at random")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("alert budget (% of the day's departures)")
        ax.set_ylabel("delay minutes reached (%)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        at10 = curve[np.isclose(curve["budget"], 0.10)].iloc[0]
        alt_text("The comparison chart", (
            "A line chart with the alert budget on a logarithmic horizontal "
            "axis and the share of all delay minutes reached on the vertical "
            "axis. Three lines: this model, a lookup table of each route's "
            "historical late rate, and random selection.\n\n"
            f"At a 10% budget the model reaches "
            f"{at10['model_delay_min_share']:.1%} of delay minutes, the "
            f"historical lookup reaches {at10['hist_delay_min_share']:.1%}, "
            f"and random selection reaches "
            f"{at10['random_delay_min_share']:.1%}. The model's advantage over "
            "random is roughly 2.4 times, and over the historical lookup "
            "roughly 1.6 times. All three converge at a 100% budget, where "
            "everything is alerted on and ranking is irrelevant."))

        with technical():
            st.markdown(f"""
| Assumption | Value | Source |
|---|---|---|
| Aircraft operating cost | ${a['cost_per_block_minute_usd']}/block minute | Airlines for America, DOT Form 41, 2025 |
| Passenger value of time | ${a['passenger_value_of_time_usd_per_hour']}/hour | FAA benefit-cost guidance |
| Load factor | {a['load_factor']:.1%} | BTS domestic, 2013 |
| Desk cost per alert | ${a['cost_per_alert_usd']:.0f} | our assumption, swept |
| CO₂ per delay minute | {a['co2_kg_per_delay_minute']:.0f} kg | derived; order of magnitude |

Break-even sits at **{sens['breakeven_effectiveness_total']:.2%}** mitigation
effectiveness — so low that cost is not what gates this system. The more
legible form of the same statement: a desk could spend up to
**${sens['breakeven_cost_per_alert_usd']:,.0f} per alert** before the
arithmetic turns negative, against roughly $6 of actual handling time. Each
alert carries **{sens['delay_min_per_alert']:.0f} delay minutes** on average.

The binding question is therefore not whether the system pays for itself. It is
whether advance warning changes an outcome at all — which this dataset cannot
answer, and which no honest analysis of it should claim to.

Total exposure in the held-out period: {e['total_delay_minutes']:,.0f} delay
minutes across {e['total_late']:,} late flights, or
{e['total_passenger_delay_hours']:,.0f} passenger-hours.
""")


# ---------------------------------------------------------------------------
# Tab 4 -- who it reaches
# ---------------------------------------------------------------------------

with tab_fair:
    st.markdown("### Who the alerts reach, and who they skip")
    fair = load_metric("fairness.json")

    if fair is None:
        st.info("Run `make fairness` to generate "
                "`reports/metrics/fairness.json`, then reload this page.")
    else:
        st.markdown(
            '<p class="fr-step">A ranking model with a fixed budget is a '
            'rationing device. "Highest probability first" is efficient, but '
            'it is not neutral: it hands the budget to whichever groups '
            'already run late most often. This page measures where it lands '
            'instead of assuming that is fine.</p>', unsafe_allow_html=True)

        which = st.radio("Group flights by",
                         ["carrier", "destination_size", "aircraft_size",
                          "time_of_day", "origin"],
                         horizontal=True,
                         format_func=lambda s: s.replace("_", " "),
                         help="Each option splits the same held-out flights a "
                              "different way and re-reads the same alerts.")

        tab = pd.DataFrame(fair["groups"][which])
        disp = fair["disparity"][which]

        fig, ax = styled_fig((9.2, 0.55 * len(tab) + 2.2))
        t = tab.sort_values("base_rate")
        y = np.arange(len(t))
        ax.barh(y - 0.2, t["base_rate"], height=0.38, color=BLUE,
                label="ran late")
        b2 = ax.barh(y + 0.2, t["recall"], height=0.38, color=ORANGE,
                     label="of those, we warned")
        for bar in b2:
            bar.set_hatch("//")
        ax.set_yticks(y)
        ax.set_yticklabels(t["group"])
        ax.set_xlabel("share of the group's flights")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        st.pyplot(fig)

        worst = t.loc[t["recall"].idxmin()]
        best = t.loc[t["recall"].idxmax()]
        alt_text("The coverage chart", (
            f"A grouped horizontal bar chart, one pair of bars per "
            f"{which.replace('_', ' ')}. The first bar is the share of that "
            f"group's flights that ran more than 15 minutes late; the second, "
            f"hatched, is the share of those late flights the alert budget "
            f"warned about.\n\n"
            f"Best served: **{best['group']}** — {best['base_rate']:.1%} ran "
            f"late and {best['recall']:.1%} of those were warned about. "
            f"Worst served: **{worst['group']}** — {worst['base_rate']:.1%} "
            f"ran late but only {worst['recall']:.1%} of those were warned "
            f"about. The gap between the two is "
            f"{disp['recall_gap']:.1%} of late flights."))

        st.markdown(
            f'<div class="fr-card">The widest coverage gap is '
            f'<strong>{disp["recall_gap"]:.1%}</strong>: '
            f'<strong>{disp["lowest_recall_group"]}</strong> gets the least '
            f'attention relative to how often it actually runs late. The model '
            f'also under-predicts everywhere, worst for '
            f'<strong>{disp["worst_calibrated_group"]}</strong> '
            f'({disp["max_abs_calibration_error"]:.1%} off).</div>',
            unsafe_allow_html=True)

        price = fair["price_of_equity"].get(which)
        if price:
            direction = ("costs" if price["delay_min_cost_of_equity"] > 0
                         else "gains")
            st.markdown("#### What it would cost to even that out")
            st.markdown(
                f"Give every group its proportional share of the *same* daily "
                f"budget, ranking by risk within each group. The coverage gap "
                f"falls from **{price['global_recall_gap']:.1%}** to "
                f"**{price['proportional_recall_gap']:.1%}**, and the desk "
                f"{direction} "
                f"**{abs(price['delay_min_cost_of_equity']):,.0f} delay "
                f"minutes** — "
                f"{abs(price['delay_min_cost_of_equity_pct']):.1f}% of what it "
                f"was reaching before, on an identical "
                f"{price['proportional_alerts']:,} alerts.")
            if price["delay_min_cost_of_equity"] < 0:
                st.success(
                    "Here fairness is not a trade-off: spreading the budget "
                    "reaches **more** delay, not less. That is a symptom, not "
                    "a free lunch — it means the global ranking is "
                    "mis-calibrated across these groups, and evening it out "
                    "corrects the error.")

        with technical():
            st.dataframe(
                tab[["group", "n", "base_rate", "share_of_flights",
                     "share_of_alerts", "recall", "fpr", "precision",
                     "calibration_error"]].style.format({
                         "base_rate": "{:.1%}", "share_of_flights": "{:.1%}",
                         "share_of_alerts": "{:.1%}", "recall": "{:.1%}",
                         "fpr": "{:.1%}", "precision": "{:.1%}",
                         "calibration_error": "{:+.3f}"}),
                hide_index=True, width="stretch")
            st.caption(
                "`recall` is the equal-opportunity criterion. "
                "`calibration_error` is mean predicted risk minus the actual "
                "rate, so negative means the model under-warns that group. "
                f"Groups with fewer than {fair['min_group_size']} flights are "
                "omitted, because a rate over a handful of flights is noise.")


# ---------------------------------------------------------------------------
# Tab 5 -- model card
# ---------------------------------------------------------------------------

with tab_model:
    ctx = joblib.load(MODELS / "training_context.joblib")
    st.markdown("### Model card")
    st.markdown(f"""
**What it does** Estimates the probability that a flight leaving JFK, LaGuardia
or Newark arrives more than 15 minutes late — the FAA and BTS on-time
definition.

**When it decides** At the scheduled departure time, before push-back. The
observed departure delay, actual departure time, air time and arrival time are
all withheld. A model given those scores far higher (PR-AUC 0.846 against
0.507) and is useless for planning, because by the time you know them the
decisions have already been made.

**Who it is for** An airline operations desk or an airport duty manager with
more flights than staff. Not for passengers deciding whether to leave for the
airport — the calibration is not tight enough at the individual-flight level
for that, and being told "38%" is not actionable advice.

**Data** NYC Flights 2013 — 336,776 departures, 327,346 of them labelled.
Joined to hourly airport weather, the FAA aircraft registry, airport metadata
and carrier names.

**Split** Temporal, never random. Train Jan–Aug ({len(D['train']):,} flights),
validate Sep–Oct ({len(D['valid']):,}), test Nov–Dec ({len(test):,}).

**Model** LightGBM, {len(ctx['features_preflight'])} pre-departure features,
hyperparameters chosen by a 40-draw random search over forward-chaining
time-series folds inside the training period.

**Held-out performance** ROC-AUC 0.716, PR-AUC 0.507 against a 25.0% base rate.
At the cost-optimal threshold of 0.20: precision 47%, recall 49%.

**The worse the outcome, the better it is predicted**

| Outcome | Base rate | ROC-AUC | Lift in riskiest 10% |
|---|---|---|---|
| Late > 15 min | 25.0% | 0.716 | 2.6× |
| Late > 60 min | 7.4% | 0.770 | 4.1× |
| Late > 120 min | 2.3% | 0.793 | 5.1× |
| Cancelled | 2.3% | 0.936 | 8.0× |
| Diverted | 0.3% | 0.608 | 2.0× |

Ranking by cancellation risk puts 80% of all December cancellations in the top
10% of the list. Diversion is the honest failure: it is decided in the air by
conditions at the destination, and this dataset has weather for the three New
York origins only.

**How far ahead it works** Replacing each flight's weather with the observation
from *h* hours earlier — a persistence forecast, the crudest kind — costs almost
nothing at three hours (PR-AUC 0.507 to 0.487) and still retains 72% of
weather's contribution at six. At 24 hours it lands exactly on the no-weather
floor.

**Known limitations**

- December is systematically under-predicted. A rolling 14-day recalibration
  cuts the Brier score from 0.180 to 0.170 and is the recommended fix;
  retraining is not.
- Coverage is uneven across carriers — see *Who it reaches*. Ranking purely by
  probability under-serves airlines whose delays the model reads less well.
- Diversions are not predictable from NYC-origin features.
- The horizon curve uses persistence, so it is a lower bound; a real forecast
  feed would land above it, by an unknown margin.
- Only NYC departures are recorded, so the true inbound leg of each aircraft is
  partly invisible and delay propagation is only partly observable.
- The data is from 2013. Fleet mixes, schedules and ATC procedures have moved
  on; the method transfers, the fitted coefficients do not.

**Not to be used for** Denying boarding, pricing a ticket, or any decision
about an individual passenger. It ranks flights, and it was never validated
for anything else.
""")
    with technical("Chosen hyperparameters"):
        st.json(ctx["best_params"])
