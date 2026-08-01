"""Generate and execute notebooks/01_walkthrough.ipynb.

Building the notebook from a script rather than editing JSON by hand keeps it
in sync with the library code: every cell calls the same functions the pipeline
uses, so the notebook cannot drift away from the results in the report.

    python scripts/build_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_walkthrough.ipynb"

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

CELLS = [
    MD("""# FlightRisk NYC — walkthrough

Predicting whether a flight leaving JFK, LGA or EWR will arrive more than 15
minutes late, **at the scheduled departure time, before push-back**.

This notebook runs the same code paths as the command-line pipeline. Run
`make reproduce` first so the cached splits and fitted models exist."""),

    MD("## 1. Load the raw tables"),
    CODE("""import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
from IPython.display import Image, display

from src.data_loader import load_tables

tables = load_tables()
pd.DataFrame({
    "table": list(tables),
    "rows": [len(t) for t in tables.values()],
    "columns": [t.shape[1] for t in tables.values()],
})"""),

    CODE("""tables["flights"].head(3)"""),

    MD("""## 2. The decision that defines the project

`dep_delay` is in the flights table and correlates with `arr_delay` at about
0.9. It is also unknown until the aircraft actually moves, so a model that uses
it cannot inform any pre-departure decision."""),
    CODE("""f = tables["flights"].dropna(subset=["arr_delay", "dep_delay"])
print(f"correlation of dep_delay with arr_delay: {f.dep_delay.corr(f.arr_delay):.3f}")
print(f"share of flights arriving >15 min late:  {(f.arr_delay > 15).mean():.1%}")"""),

    MD("## 3. Build the features and the temporal split"),
    CODE("""from src.pipeline import load_splits, xy
from src.config import MODE_A, MODE_B

train, valid, test, manifest = load_splits()
pd.DataFrame({
    "split": ["train (Jan-Aug)", "valid (Sep-Oct)", "test (Nov-Dec)"],
    "flights": [len(train), len(valid), len(test)],
    "late rate": [f"{d.is_delayed.mean():.1%}" for d in (train, valid, test)],
})"""),

    MD("""The validation base rate sits ten points below both neighbours. That is
not a mistake in the split — September and October 2013 really were unusually
smooth — and it drives the calibration result in section 8."""),

    CODE("""feats = manifest["features"][MODE_A]
print(f"{len(feats)} pre-flight features")
from src.explain import feature_groups
for name, cols in feature_groups(feats).items():
    print(f"  {name:18s} {len(cols):2d}  e.g. {', '.join(cols[:3])}")"""),

    MD("""### Leakage checks

These run as `pytest tests -q`. The two that matter most: the inbound-leg
delay must never come from the future, and the tail-number target encoding must
be genuinely out-of-fold."""),
    CODE("""from sklearn.metrics import roc_auc_score
from src.config import LEAKY_COLUMNS, GATE_ONLY_FEATURES

assert not set(LEAKY_COLUMNS) & set(feats)
assert not set(GATE_ONLY_FEATURES) & set(feats)
auc = roc_auc_score(train.is_delayed, train.te_tailnum)
print(f"te_tailnum alone, in training: AUC {auc:.3f}  (a leaked encoding approaches 1.0)")"""),

    MD("## 4. What the data says"),
    CODE("""display(Image("../reports/figures/02_temporal_patterns.png"))
display(Image("../reports/figures/04_weather_effects.png"))"""),

    CODE("""full = pd.concat([train, valid, test])
pd.DataFrame({
    "dry hours":            [f"{full.loc[full.wx_precip == 0, 'is_delayed'].mean():.1%}"],
    "any precipitation":    [f"{full.loc[full.wx_precip > 0, 'is_delayed'].mean():.1%}"],
    "visibility >= 3 mi":   [f"{full.loc[full.wx_visib >= 3, 'is_delayed'].mean():.1%}"],
    "visibility < 3 mi":    [f"{full.loc[full.wx_visib < 3, 'is_delayed'].mean():.1%}"],
}, index=["late rate"]).T"""),

    MD("""### The rotation feature we had to fix

The dataset records only *departures* from NYC, so an aircraft's return leg into
New York is invisible. Measuring turnaround as the gap since the previous leg's
scheduled arrival therefore spans an entire unobserved round trip. Subtracting
the previous block time recovers the real schedule slack."""),
    CODE("""bands = pd.cut(full.rotation_slack_min, [-1e9, 0, 45, 90, 180, 1e9],
               labels=["negative", "0-45 m", "45-90", "90-180", "180 m+"])
full.groupby(bands, observed=True).agg(
    late_rate=("is_delayed", "mean"), flights=("is_delayed", "size")).round(3)"""),

    MD("## 5. Held-out results"),
    CODE("""results = json.load(open("../reports/metrics/evaluation.json"))
rows = []
for k in ["prior", "historical_rate", "logistic_regression", "random_forest",
          "xgboost", "lightgbm", "lightgbm_gate"]:
    t = results[k]["test"]
    rows.append({"model": k, "PR-AUC": round(t["pr_auc"], 4),
                 "ROC-AUC": round(t["roc_auc"], 4), "Brier": round(t["brier"], 4)})
pd.DataFrame(rows).set_index("model")"""),

    CODE("""display(Image("../reports/figures/08_roc_pr_curves.png"))
display(Image("../reports/figures/13_prediction_horizon.png"))"""),

    MD("""Adding the observed departure delay lifts PR-AUC from 0.507 to 0.846.
Most of the apparent skill of a "flight delay model" is the observation that the
plane left late — which is not available when the decision has to be made."""),

    MD("## 6. From probability to decision"),
    CODE("""d = results["decision"]
pd.Series({
    "threshold chosen on validation": d["threshold_chosen_on_validation"],
    "precision at that threshold":    round(d["test_at_chosen_threshold"]["precision"], 3),
    "recall at that threshold":       round(d["test_at_chosen_threshold"]["recall"], 3),
    "cost per flight":                round(d["test_at_chosen_threshold"]["cost_per_flight"], 3),
    "cost at the default 0.5":        round(d["cost_at_default_0.5"], 3),
    "cost of alerting on nothing":    round(d["cost_of_alerting_nothing"], 3),
    "cost of alerting on everything": round(d["cost_of_alerting_everything"], 3),
}).to_frame("value")"""),

    CODE("""c = results["capacity"]
print(f"Top {c['k']:.0%} of flights by predicted risk: {c['n_flights_flagged']:,} flights")
print(f"  precision {c['precision_at_k']:.1%}  |  lift {c['lift']:.2f}x over the base rate")"""),

    MD("## 7. What the model uses, and what it is worth"),
    CODE("""display(Image("../reports/figures/16_shap_importance.png"))"""),

    CODE("""abl = json.load(open("../reports/metrics/ablation.json"))
full_ap = abl["full model"]["test_pr_auc"]
pd.DataFrame([
    {"family removed": k, "test PR-AUC": round(v["test_pr_auc"], 4),
     "change": round(v["test_pr_auc"] - full_ap, 4)}
    for k, v in abl.items() if k != "full model"
]).sort_values("change").set_index("family removed")"""),

    MD("""Weather is worth ten times more than anything else. The eight historical-rate
target encodings — which SHAP credits with 24% of influence — are worth nothing:
removing them improves the held-out score. Attribution measures what the fitted
model leans on; only an ablation measures what the data is worth."""),

    MD("## 8. Calibration under seasonal shift"),
    CODE("""display(Image("../reports/figures/15_calibration_drift.png"))
r = results["recalibration"]
pd.DataFrame(r["brier"], index=["Brier"]).T.join(
    pd.DataFrame(r["mean_prediction_vs_actual"], index=["mean prediction"]).T
).round(4)"""),

    MD("""The model tracks November closely and under-predicts December by a wide
margin. The cause is `day_of_year`: every test value lies outside the training
range, so it applies a near-constant −0.35 log-odds offset to all 54,000 test
flights. Refitting an isotonic calibrator on a trailing 14-day window each day
cuts the Brier score by 5.7% and pulls the mean prediction from 0.183 to 0.259
against an actual 0.276.

**For this problem the maintenance burden is recalibration, not retraining.**"""),

    MD("## 9. Explaining one flight"),
    CODE("""import joblib, shap
from src.config import MODELS
from src.explain import pretty

clf = joblib.load(MODELS / "lightgbm.joblib")
X, y = xy(test, feats)
p = clf.predict_proba(X)[:, 1]

i = int(np.argmax(p))
row = test.iloc[i]
print(f"{row.carrier}{int(row.flight)}  {row.origin}->{row.dest}  {row.flight_date.date()}")
print(f"predicted risk {p[i]:.0%}  |  actually arrived {row.arr_delay:+.0f} min")

sv = shap.TreeExplainer(clf).shap_values(X.iloc[[i]])
sv = sv[1] if isinstance(sv, list) else sv
s = pd.Series(sv[0], index=X.columns)
s = s.reindex(s.abs().sort_values(ascending=False).index[:8])
s.index = [pretty(c) for c in s.index]
s.to_frame("effect on log-odds").round(3)"""),

    MD("""## 10. The worse the outcome, the better it is predicted

The clearest pattern in the project. It turned up three separate times."""),
    CODE("""sev = json.load(open("../reports/metrics/severity_v2.json"))
dis = json.load(open("../reports/metrics/disruption.json"))

rows = []
for k, label in [("gt15", "late > 15 min"), ("gt60", "late > 60 min"),
                 ("gt120", "late > 120 min")]:
    t = sev["tiers"][k]
    rows.append({"outcome": label, "base rate": f"{t['base_rate']:.1%}",
                 "ROC-AUC": round(t["roc_auc"], 3),
                 "lift in riskiest 10%": f"{t['lift_at_10pct']:.1f}x"})
for k, label in [("is_cancelled", "cancelled"), ("is_diverted", "diverted")]:
    t = dis[k]
    rows.append({"outcome": label, "base rate": f"{t['base_rate_test']:.1%}",
                 "ROC-AUC": round(t["roc_auc"], 3),
                 "lift in riskiest 10%": f"{t['lift_at_10pct']:.1f}x"})
pd.DataFrame(rows).set_index("outcome")"""),

    MD("""Severe disruption has causes that sit in the feature set — storms,
closed runways, broken rotations. Marginal lateness is mostly noise. The
cancellation model, trained on the 9,430 flights the main model *discarded* as
unlabellable, is the strongest of the lot: it puts 80% of December's
cancellations in the top 10% of the ranked list.

Diversion is the honest failure. It is decided in the air by conditions at the
destination, and this dataset has weather for the three NYC origins only."""),
    CODE("""display(Image("../reports/figures/26_disruption_model.png"))"""),

    MD("""### Quantile heads tell the same story

The P90 head beats a constant by 16.4%; the P50 head manages 2.0%, i.e.
nothing. The tail is predictable, the centre is not."""),
    CODE("""q = sev["quantile"]
pd.DataFrame({
    k.upper(): {"pinball": round(v["pinball"], 3),
                "constant baseline": round(v["pinball_constant_baseline"], 3),
                "improvement": f"{v['pinball_improvement_pct']:.1f}%",
                "coverage": f"{v['coverage']:.3f} (target {v['coverage_target']:.2f})"}
    for k, v in q.items()})"""),

    MD("""## 11. How far ahead does it work?

Each flight is given the weather observation from *h* hours before its
scheduled departure and the model retrained. That is a persistence forecast —
the crudest kind — so the curve is a **lower bound** on a real forecast-fed
model."""),
    CODE("""hor = json.load(open("../reports/metrics/horizon.json"))
abl = json.load(open("../reports/metrics/ablation.json"))
floor = abl["weather"]["test_pr_auc"]
h0 = hor["0"]["pr_auc"]

pd.DataFrame([{
    "horizon": f"{int(k)} h",
    "PR-AUC": round(v["pr_auc"], 4),
    "precision in riskiest 10%": f"{v['precision_at_10pct']:.1%}",
    "weather value retained": f"{(v['pr_auc'] - floor) / (h0 - floor):.0%}",
} for k, v in sorted(hor.items(), key=lambda kv: int(kv[0]))]).set_index("horizon")"""),

    CODE("""display(Image("../reports/figures/25_forecast_horizon.png"))
print(f"model trained with no weather at all: {floor:.4f}")
print(f"model trained on 24-hour-old weather:  {hor['24']['pr_auc']:.4f}")"""),

    MD("""Three hours costs 0.020 PR-AUC and keeps 86% of what weather
contributes, so the system is deployable well before push-back. Twenty-four
hours lands *exactly* on the no-weather floor — yesterday's weather at this
hour is worth nothing, which is a built-in check that the experiment measures
what it claims."""),

    MD("""## 12. Try it interactively

```bash
make app
```

Single-flight risk with its explanation and a severity panel (P90 worst case,
risk of >60 min and >2 h), an operations-desk view that ranks a whole day
against a fixed alerting budget, and a model card with the limitations. Full
write-up in `reports/report.md`."""),
]


def build(execute: bool = True) -> None:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}

    if execute:
        client = NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT / "notebooks")}})
        client.execute()

    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(nb.cells)} cells, executed={execute})")


if __name__ == "__main__":
    sys.exit(build(execute="--no-exec" not in sys.argv))
