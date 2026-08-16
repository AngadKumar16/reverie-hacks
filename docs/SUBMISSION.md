# ReverieHacks 2026 — Datathon submission

**FlightRisk NYC — predicting arrival delay before the aircraft moves**

Deadline: 24 August 2026, 12:00am CDT · Track: **Datathon** · Team size: 1–3

---

## The four required files

The Datathon track requires four things. Here is each one, and where it is.

### 1. Code repository

<https://github.com/aaravsinhaofficial/reverie-hacks>

Public, MIT-licensed, runnable from a clean clone in two commands:

```bash
pip install -r requirements.txt
make reproduce        # ~15 min on 4 cores: data → tests → training → every figure and metric
make app              # the interactive demo at http://localhost:8501
```

Every stage is also runnable alone (`make train`, `make fairness`, `make
transfer`, …) and the full list is in the README. `make verify` runs 119 checks
including re-reading every headline number in the report from the artefact that
produced it. CI runs the fresh-clone path on Python 3.10 and 3.12 on every push.

### 2. Dataset

**NYC Flights 2013** — <https://www.kaggle.com/datasets/aephidayatuloh/nyc-flights-2013>

Linked at the top of the README with three mirrors and provenance back to the
primary sources (US Bureau of Transportation Statistics on-time performance,
the FAA aircraft registry, ASOS/NOAA hourly weather). 336,776 flights, 327,346
labelled, five relational tables.

**No Kaggle account is needed.** `pip install -r requirements.txt` pulls the
`nycflights13` package, which ships the identical tables, and `make data`
materialises them to `data/raw/`. Kaggle CSVs dropped in that directory are
preferred automatically if present.

### 3. Report

[`reports/report.md`](../reports/report.md) — also typeset as
[`reports/report.pdf`](../reports/report.pdf).

~1,400 lines across 12 sections and 3 appendices: methodology, the feature
boundary and why it is drawn where it is, experimental design, results,
calibration under distribution shift, what the model actually uses, limitations,
what it is worth, who it reaches, deployment, and conclusions. Every figure it
references is generated; every number it quotes is re-read from
`reports/metrics/` by `make verify`, which fails if the prose has drifted.

### 4. Demo video

[`docs/demo.mp4`](demo.mp4) — 84 s, captioned, built by `make video` from the
generated figures and metrics. Every number on screen is read from
`reports/metrics/` at render time rather than typed into a slide, so the video
cannot disagree with the report.

[`docs/DEMO_VIDEO.md`](DEMO_VIDEO.md) is the shot-by-shot script for a live
screen recording of the app, which is the stronger submission if there is time
to record it.

---

## The one-paragraph version

Will a flight leaving JFK, LaGuardia or Newark arrive more than 15 minutes late?
This answers that **at the scheduled departure time, before push-back** — the
only moment at which the answer is still useful for planning. Most published
models on this dataset quietly include the observed departure delay and report
ROC-AUC above 0.90; we measured that shortcut rather than taking it, and the
same model on the same split scores **PR-AUC 0.846 with it and 0.507 without**.
The second number is the honest one, and the project is built around it: the
model is evaluated not as a classifier but as a rationing device for an
operations desk that can only act on a few dozen of several hundred daily
departures.

---

## Against the judging criteria

### Innovation — originality, creative approach

The original move is **the question, not the algorithm**. Flight-delay
prediction on NYC Flights 2013 is one of the most-worked datasets in data
science, and the standard result — AUC above 0.90 — is an artefact of including
`dep_delay`. We built the honest version, quantified exactly what the shortcut
is worth (0.846 vs 0.507, so two thirds of the field's apparent skill), and made
that measurement the spine of the report.

Three further findings we have not seen made on this dataset:

- **The worse the outcome, the better it is predicted.** ROC-AUC rises
  monotonically from 0.716 for a 15-minute delay to 0.770 at 60 minutes, 0.793
  at 120, and **0.936 for cancellation** — where the riskiest 10% of the list
  contains 80% of all December cancellations. Marginal lateness is mostly noise;
  severe disruption has causes that are in the feature set.
- **Feature importance is not predictive value.** SHAP ranks `day_of_year`
  first, but every test value lies outside the training range, so it applies a
  near-constant −0.35 log-odds offset to all 54,000 test flights. It moves the
  level, not the ranking, and it is responsible for the December
  under-prediction. The ablation — which measures value rather than attribution
  — ranks it near zero.
- **Retraining is not the fix for drift; recalibration is.** A rolling 14-day
  isotonic recalibration cuts the test Brier score from 0.180 to 0.170.
  Calibrating once on the validation period does nothing.

### Problem solving — relevance, effectiveness, feasibility

**Relevance.** US flight delay costs about $33 billion a year, $16.7bn of it
borne by passengers. The constraint on an ops desk is attention, not knowledge:
several hundred departures a day, a few dozen actionable.

**Effectiveness, measured against the alternatives rather than against zero.**
On a 10%-per-day alert budget:

| Rule | Precision | Delay minutes reached | Share of all delay |
|---|---:|---:|---:|
| Random | 25.0% | 75,965 | 10.0% |
| Route's historical late rate (no ML) | 31.6% | 111,426 | 14.6% |
| **This model** | **47.2%** | **183,879** | **24.2%** |

2.4× random and 1.65× the no-ML alternative that a desk actually has today.
Budget is spent per day — a desk cannot save November's alerts for 22 December
— which makes the numbers lower and the exercise honest.

**Feasibility.** The system needs a schedule feed and an hourly METAR/ASOS
weather feed, both of which already exist in every operations centre, and 5.2 MB
of model. It is a cron job, not a service. See Sustainability below for the
measured costs.

**And the honest limit.** Converting delay minutes to dollars needs a mitigation
effectiveness — how much of a warned delay a desk actually recovers — which no
public dataset measures. We never assume it: 10% for the headline, 2–40% swept,
break-even reported at 0.11%. Section 9.5 of the report is titled "What this
section is not", and says in those words that it is a ceiling on available value
rather than a measured saving.

### Sustainability & scalability

**Does it scale technically?** Measured, not asserted (`make cost`):

| | |
|---|---|
| Deployable artefacts | 5.2 MB |
| Scoring one day of NY departures | **12.5 ms** (p95 15.9 ms) |
| Scoring the whole test period | 0.60 s — 90,816 flights/s |
| 100× New York's volume | **7.2 s of compute per day**, one core |

No GPU, no vector database, no inference API bill. Building the features costs
five times more than running the model, which tells you what to optimise.

**Does it generalise?** This is the question a delay model usually dodges, so we
measured it (`make transfer`). Each airport held out in turn — trained,
historical-rate-encoded and early-stopped on the other two only, then scored on
the held-out airport's Nov–Dec flights:

| Held out | Never-seen PR-AUC | Full-network PR-AUC | Skill retained |
|---|---:|---:|---:|
| EWR | 0.492 | 0.533 | **92.4%** |
| JFK | 0.403 | 0.454 | **88.7%** |
| LGA | 0.510 | 0.535 | **95.3%** |

**A model deployed where it has no local history keeps 92% of its skill.** What
transfers is weather, congestion and rotation slack; what does not is local
history, and it turns out not to matter much. The limit is stated plainly: three
airports 20 miles apart share a weather system, so this rules out a
lookup-table failure mode but does not prove transfer to Denver.

**Environmental impact, both directions.** Recovered delay minutes are unburned
fuel: a 10% budget at 10% effectiveness avoids **331 tonnes of CO₂ over two
months**, ~2,000 tonnes a year across the three airports. Against that, training
the model costs 0.33 core-hours — **0.6–3.7 grams of CO₂**, about as much as
twelve seconds of one delayed narrowbody idling on a taxiway. Both figures are
swept rather than point-estimated.

**Social risk, named before someone finds it.** A delay-risk score is a ranking,
and rankings get reused for things they were not validated for. Section 11.4
names three foreseeable failure modes — feedback loops from systematic
under-alerting, misuse against individual passengers, and false confidence in a
47%-precision list — and the model card in the app says explicitly that nothing
here should touch boarding, pricing or compensation eligibility.

**Who the system reaches.** Ranking by probability is a rationing rule and it
has losers. Coverage of genuinely-late flights ranges from **38.2% to 1.9%**
across carriers: Southwest runs late more often than JetBlue and gets a
fortieth of the alerts, because the model under-predicts it by 17 points —
invisible in the AUC. Allocating the same budget proportionally cuts the gap to
6.4 points for 11.6% of the delay caught; across destination-size quartiles the
same change *gains* 1.3%. Both numbers are computed, not asserted, and the
finding changed the recommendation.

### User experience & design

The interface was designed rather than defaulted (`make app`). Five views: a
single flight with its reasons in plain English, an operations-desk day view
with a movable alert budget, the impact model with its assumptions exposed as
sliders, the equity audit, and a model card.

- **Colour is never the only signal.** Every risk level carries a word and a
  shape as well as a colour, and the palette is Okabe-Ito, which survives all
  three common forms of colour blindness.
- **Every chart has a generated text alternative**, not just a caption — so the
  app is usable with a screen reader rather than merely compliant.
- **Plain English is the default reading.** Statistics live behind "the
  technical version" expanders, so a non-specialist gets a sentence and a
  specialist gets the number.
- **High-contrast and larger-text switches** are in the sidebar.
- The default framing is a triage aid rather than a prediction, because a 47%
  precision list is wrong slightly more often than it is right and the interface
  should not pretend otherwise.

### Bonus: exceptionality

The thing we would point at is that **the report cannot lie about the results**.

`make verify` runs 119 checks. Beyond leakage and determinism, it re-reads every
headline number quoted in `report.md` and `README.md` from the metrics file that
generated it and fails if they disagree. It also scrambles every post-departure
column and confirms the shipped model's predictions are *exactly* unchanged, and
refits from the seed to confirm the score reproduces to six decimals and hashes
to the recorded booster fingerprint.

Reproducibility was measured rather than claimed. Every generated artefact was
deleted and the pipeline re-run from the raw tables: **1,153 metric values
compared, 1,115 identical, and all 38 differences were wall-clock timing
fields**. Then the whole thing was rebuilt on a second CPU architecture
(arm64 vs x86-64): the deployed model reproduced to **all six decimals** on the
same 391 trees, and the only drift was in BLAS-backed cross-checks, the largest
being 0.0023 of PR-AUC in XGBoost. Both results are in the README with the
numbers.

Two bugs found and fixed while preparing this submission are documented rather
than quietly patched: the dataset loader failed on Python 3.12+ because
`nycflights13` imports the removed `pkg_resources`, and `make severity` skipped
retraining on a fresh clone because it resumed from a committed metrics file
whose model artefacts are generated. Both were fresh-clone-only failures — which
is why CI now runs the fresh-clone path on every push.

---

## Team & attribution

Solo submission. Data is public-domain US government data redistributed under
the `nycflights13` package's CC0 terms; code is MIT.

AI assistance was used the way a pair programmer is used, and the guard against
that is structural rather than promissory: every number in the report has to
survive `make verify`, which re-reads it from the artefact that produced it. The
check exists precisely so nothing in the prose is taken on trust.
