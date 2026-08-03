# FlightRisk NYC — predicting arrival delay *before* the aircraft moves

Will a flight leaving JFK, LaGuardia or Newark arrive more than 15 minutes
late? This project answers that at the **scheduled departure time, before
push-back** — the only moment at which the answer is still useful for planning.

Most published flight-delay models quietly include `dep_delay`, the observed
departure delay, and report ROC-AUC above 0.90. We measured that shortcut
instead of taking it: the same model, same data, same split scores **PR-AUC
0.846 with the departure delay and 0.507 without it**. The second number is the
honest one, and it is the number this repository is built around.

| | |
|---|---|
| **Data** | [NYC Flights 2013](https://www.kaggle.com/datasets/aephidayatuloh/nyc-flights-2013) — 336,776 flights, 327,346 labelled |
| **Task** | Binary classification: arrival delay > 15 min (the FAA on-time definition) |
| **Split** | Temporal — train Jan–Aug, validate Sep–Oct, test Nov–Dec |
| **Best model** | Gradient boosting, 67 pre-flight features, 40-draw random search per library |
| **Held-out result** | ROC-AUC **0.716**, PR-AUC **0.507** (XGBoost 0.513) against a 25.0% base rate |
| **Operational result** | Top 10% riskiest flights are **64% late** — a **2.6× lift** |
| **Cancellations** | ROC-AUC **0.936** — top 10% catches **80% of all cancellations** |
| **Horizon** | 3 h ahead costs 0.020 PR-AUC; 24 h ahead is worth nothing |
| **Impact** | A 10%/day alert budget reaches **24.2% of all delay minutes** — **2.4× random**, **1.65×** a no-ML historical lookup |
| **Equity** | Coverage of late flights ranges **38.2% → 1.9%** across carriers; evening it out costs **11.6%** of the delay caught |

---

## Why this problem

US flight delay costs about **$33 billion a year** — $16.7bn of it borne by
passengers in lost time, missed connections and unplanned hotels, $8.3bn
directly by airlines.<sup>[1]</sup> A single minute of delay is $98.41 of
aircraft operating cost by the carriers' own DOT Form 41 filings,<sup>[2]</sup>
and passenger time is valued at $47/hour by the FAA's benefit-cost
guidance.<sup>[3]</sup>

An operations desk at a New York airport oversees several hundred departures a
day and can act on a few dozen. It does not need a better description of
yesterday; it needs a ranked list early enough to swap an airframe or hold a
connection. That constraint — **attention, not knowledge** — is what this
project is built around, and it is why the model is evaluated as a rationing
device on a fixed daily budget rather than as a classifier.

<sup>[1]</sup> NEXTOR *Total Delay Impact Study* for the FAA.
<sup>[2]</sup> [Airlines for America, U.S. Passenger Carrier Delay Costs, 2025](https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs/).
<sup>[3]</sup> FAA, *Economic Values for FAA Investment and Regulatory Decisions*, §1.

---

## Dataset

**Primary link:** <https://www.kaggle.com/datasets/aephidayatuloh/nyc-flights-2013>

Mirrors of the same tables:

- Kaggle, all five tables: <https://www.kaggle.com/datasets/ashwinsanthanam/nyc-flights-data-from-nycflights13-package-in-r>
- Canonical source (tidyverse R package): <https://github.com/tidyverse/nycflights13>
- CRAN documentation: <https://cran.r-project.org/web/packages/nycflights13/index.html>

Everything traces back to the same primary records: US Bureau of Transportation
Statistics on-time performance for 2013, the FAA aircraft registry, and
ASOS/NOAA hourly weather observations.

| Table | Rows | What it contributes |
|---|---|---|
| `flights` | 336,776 | one row per NYC departure: schedule, route, carrier, tail number, delays |
| `weather` | 26,115 | hourly observations at EWR/JFK/LGA — temp, wind, gust, visibility, precipitation, pressure |
| `planes` | 3,322 | aircraft registry: build year, seats, engines, manufacturer |
| `airports` | 1,458 | coordinates, elevation, timezone (needed to compute block time correctly) |
| `airlines` | 16 | carrier code → name |

**You do not need a Kaggle account to run this.** `pip install -r
requirements.txt` pulls the `nycflights13` package, which ships the identical
tables, and `make data` writes them to `data/raw/`. If you prefer the Kaggle
download, drop `flights.csv`, `weather.csv`, `planes.csv`, `airports.csv` and
`airlines.csv` into `data/raw/` and the loader will prefer them automatically.

---

## Quick start

```bash
git clone <this-repo> && cd flightrisk-nyc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make reproduce        # data -> tests -> EDA -> training -> evaluation -> SHAP
make app              # interactive demo at http://localhost:8501
```

`make reproduce` takes roughly 15 minutes on 4 cores and regenerates every
figure and metric quoted in the report. Each stage is also runnable on its own:

```bash
make data       # build data/raw CSVs + cached feature splits
make test       # 16 leakage and correctness checks — run these first
make eda        # figures 01–07
make train      # baselines + 40-draw random search + final fits (resumable)
make evaluate   # figures 08–15, reports/metrics/evaluation.json
make explain    # figures 16–22, SHAP + feature-family ablation
make severity   # figures 23–24, severity tiers + quantile heads
make horizon    # figure 25, forecast-horizon degradation curve
make disruption # figure 26, cancellation / diversion / disruption models
make impact     # figures 27-28, delay minutes / passenger hours / $ / CO2
make fairness   # figure 29, who the alert budget reaches
make verify     # determinism, leakage, report-vs-artefact agreement
```

`make verify` is worth calling out. Beyond the unit tests it confirms that the
feature pipeline rebuilds bit-identically from the raw tables, that a fresh
refit from the same seed reproduces the reported score to six decimals and
hashes to the recorded booster fingerprint, that scrambling every
post-departure column leaves the shipped model's predictions *exactly*
unchanged, and that **every headline number in the report is re-read from the
metrics files and compared** — so the prose cannot drift away from the results.

### Reproducibility, actually tested

Claiming determinism is cheap, so this was measured. Every generated artefact
(`data/raw`, `data/processed`, `models`, `reports/figures`, `reports/metrics`)
was deleted and the whole pipeline re-run from the raw tables:

| Check | Result |
|---|---|
| Leaf values across all metric JSON files | **1,153 compared, 1,115 identical** |
| Values that differed | **38 — every one a wall-clock `seconds` timing field** |
| Differing values excluding timings | **0** |
| All 40 LightGBM search draws | bit-identical CV scores |
| All 8 XGBoost search draws (of the 8 run at the time) | bit-identical CV scores |
| Parquet splits, `training_context`, RF / logistic / XGBoost pickles | byte-identical |
| LightGBM booster SHA-256 | `32235056eebd…` before and after |

One honest caveat: a joblib pickle of an `LGBMClassifier` is *not* byte-stable
even when training is fully deterministic — the container carries incidental
state. The booster's own serialised model string **is** stable, so
`reports/metrics/model_fingerprints.json` records its SHA-256 and `make verify`
checks it. That is the checksum to compare after a rebuild.

Determinism holds on identical library versions (pinned in
`requirements.txt`). Different LightGBM or NumPy builds may shift the last
decimal places; the `approx` tolerances in `scripts/verify.py` are set for the
pinned versions.

Training is checkpointed: `make train` can be interrupted and re-run without
losing progress, and re-running a completed step is a no-op.

---

## What the model is allowed to know

This is the design decision the whole project rests on.

**`preflight` (the deployable model).** The instant of scheduled departure,
before push-back. Available: the published timetable, route, assigned airframe,
current observed weather at the NYC origin, and statistics learned from *past*
training-period flights.

**Forbidden in that mode:** `dep_delay`, `dep_time`, `air_time`, `arr_time` —
every one of them is determined at or after the moment we are predicting.

**`gate` (diagnostic only).** A few minutes later, once the aircraft has pushed
back. Adds the observed departure delay and the inbound leg's arrival delay,
the latter guarded so it is only used when that leg genuinely landed before our
scheduled departure. This mode exists to *measure* the shortcut, not to take
it.

`tests/test_features.py` enforces the boundary — including a check that the
inbound-leg feature never reads from the future, and that the tail-number
target encoding is genuinely out-of-fold rather than a copy of the label.

---

## Repository layout

```
src/
  config.py       every constant: split months, cost ratio, seeds, thresholds
  data_loader.py  loads the five tables from CSV or the packaged source
  features.py     leakage-safe feature engineering — the core of the project
  pipeline.py     builds and caches the train/valid/test splits
  models.py       baselines, sklearn pipelines, search spaces
  train.py        resumable random search + final fits
  evaluate.py     held-out metrics, calibration, cost-based thresholds
  explain.py      SHAP attribution + feature-family ablation
  eda.py          exploratory figures
  severity.py     severity tiers, quantile heads, conditional-on-late
  horizon.py      persistence-forecast horizon curve
  cancellations.py  three-outcome disruption model over all 336,776 flights
  impact.py       delay minutes -> passenger hours -> dollars -> CO2, all swept
  fairness.py     per-group coverage audit and the priced equity trade-off
tests/            16 leakage and correctness checks
app/              Streamlit demo (5 views, colour-blind-safe, text alternatives)
docs/PITCH.md     three-minute demo script and the questions we expect
notebooks/        end-to-end walkthrough
reports/
  report.md       full methodology, results and analysis
  figures/        29 generated figures
  metrics/        every number in the report, as JSON/CSV
```

---

## Headline results

All figures are on the untouched Nov–Dec 2013 test period (53,991 flights,
25.0% late).

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Base rate (no model) | 0.250 | 0.500 | 0.188 |
| Historical-rate rule | 0.340 | 0.621 | 0.182 |
| Logistic regression | 0.478 | 0.708 | 0.166 |
| Random forest | 0.491 | 0.713 | 0.164 |
| LightGBM (tuned) | 0.507 | 0.716 | 0.167 |
| **XGBoost (tuned)** | **0.513** | **0.719** | 0.168 |
| *LightGBM, post-push-back* | *0.846* | *0.903* | *0.097* |

### The worse the outcome, the better it is predicted

The clearest pattern in the project, and it turned up in three independent
experiments:

| Outcome | Base rate | ROC-AUC | Lift in riskiest 10% |
|---|---:|---:|---:|
| Late > 15 min | 25.0% | 0.716 | 2.6× |
| Late > 60 min | 7.4% | 0.770 | 4.1× |
| Late > 120 min | 2.3% | 0.793 | 5.1× |
| **Cancelled** | 2.3% | **0.936** | **8.0×** |
| Diverted | 0.3% | 0.608 | 2.0× |

Severe disruption has causes that are in the feature set — storms, closed
runways, broken rotations. Marginal lateness is mostly noise. Ranking by
cancellation risk puts **80% of all December cancellations in the top 10%** of
the list. Diversion is the honest failure: it is decided in the air by
conditions at the destination, which this dataset does not contain.

### How far ahead it works

Replacing each flight's weather with the observation from *h* hours earlier — a
persistence forecast, the crudest kind, so a **lower bound** on a real
forecast-fed model:

| Horizon | 0 h | 1 h | 2 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|---|---|
| PR-AUC | 0.507 | 0.504 | 0.492 | 0.487 | 0.467 | 0.420 | 0.363 |
| Weather value retained | 100% | 98% | 89% | 86% | 72% | 39% | 0% |

A three-hour planning horizon is nearly free. Twenty-four hours lands exactly
on the no-weather floor (0.3634 against 0.3640) — a built-in check that the
experiment measures what it claims.

Four findings the report develops:

1. **Weather is the single most valuable data source.** Removing the weather
   family costs 0.141 PR-AUC — an order of magnitude more than any other
   group. Removing the historical-rate encodings costs *nothing*.
2. **Most of the "arrival delay" signal is just departure delay.** Adding
   `dep_delay` lifts PR-AUC from 0.507 to 0.846. Papers that report the higher
   number are answering an easier question.
3. **Feature importance is not predictive value.** SHAP ranks `day_of_year`
   first, but every test value lies outside the training range, so it applies a
   near-constant −0.35 log-odds offset to all 54,000 test flights — it moves
   the level, not the ranking, and it is responsible for the December
   under-prediction.
4. **Retraining is not the fix for drift; recalibration is.** A rolling 14-day
   isotonic recalibration cuts the test Brier score from 0.180 to 0.170 and
   pulls the mean prediction from 0.183 up to 0.259 against a 0.276 actual
   rate. Calibrating once on the validation period does nothing.

---

### What a 10% alert budget is worth

The model is only useful if it beats the alternatives an operations desk
already has, so it is scored against them rather than against zero. Budget is
spent **per day** — a desk cannot save November's alerts for 22 December —
which makes the numbers lower and the exercise honest.

| Rule | Precision | Delay minutes reached | Share of all delay |
|---|---:|---:|---:|
| Random | 25.0% | 75,965 | 10.0% |
| Route's historical late rate (no ML) | 31.6% | 111,426 | 14.6% |
| **This model** | **47.2%** | **183,879** | **24.2%** |

At a deliberately pessimistic 10% mitigation effectiveness that is **$3.05M**
of recovered value over the two held-out months — $1.81M of airline operating
cost, 27,015 passenger-hours returned, 331 tonnes of CO₂ — of which **$1.69M is
attributable to the model** rather than to the budget existing at all. Every
external constant is swept and the sign never changes. Break-even is 0.11%
effectiveness, which is the real finding: cost is not what gates this system,
and whether advance warning changes an outcome is a question no historical
dataset can answer.

### Who the alerts reach

Ranking by probability is a rationing rule, and it has losers. Coverage of
genuinely-late flights ranges from **38.2% to 1.9%** across carriers. Southwest
runs late more often than JetBlue and gets a fortieth of the alerts, because
the model under-predicts it by 17 points — invisible in the AUC. Spending the
same budget proportionally cuts the gap to 6.4 points for 11.6% of the delay
caught; across destination-size quartiles it *gains* 1.3%. Both numbers are
computed, not asserted (`make fairness`).

---

## Demo

```bash
make app
```

Five views: a single flight with its reasons in plain English, an
operations-desk day view with a movable alert budget, the impact model with its
assumptions exposed as sliders, the equity audit, and a model card.

The interface was designed rather than defaulted. Colour is never the only
signal — every risk level carries a word and a shape, and the palette is
Okabe-Ito, which survives all three common forms of colour blindness. Every
chart has a generated text alternative, not just a caption. Statistics live
behind "the technical version" expanders so the default reading is plain
English. High-contrast and larger-text switches are in the sidebar.

---

## Licence

MIT — see `LICENSE`. The NYC Flights 2013 data is public-domain US government
data redistributed under the terms of the `nycflights13` package (CC0).
