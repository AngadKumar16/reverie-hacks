# Demo video — recording guide

The Datathon track requires a demo video. There are two in this repository and
they are for different things:

| | What it is | When to use it |
|---|---|---|
| **`docs/demo.mp4`** | 84 s, silent, captioned. Built by `make video` from the figures and metrics — every number on screen is read from `reports/metrics/` at render time, so it cannot disagree with the report. | Ships in the repo. A working submission on its own if recording falls through. |
| **A screen recording** | ~3 min of the live app with narration, using the script below. | The better submission. Judges want to see the thing run. |

Record the screencast if there is any time at all. This page is the shot list
for it. The three-minute judging pitch and the Q&A prep live in
[`PITCH.md`](PITCH.md); this one is about production.

---

## Before you record

```bash
make data && make train      # if models/ is empty — about 12 minutes
make app                     # http://localhost:8501
```

- **Window**: 1920×1080, browser at 100% zoom, bookmarks bar hidden. The app
  is responsive but the day view wants the width.
- **Have open**: the app on tab **1 · One flight**, and `reports/report.pdf` in
  a second window for the closing shot.
- **Set the date to 22 December 2013** before you start rolling. It is the
  busiest disrupted day in the held-out period, so every view has something to
  show.
- **Turn on Do Not Disturb.** Notification banners are the most common reason a
  take gets thrown away.
- **Say the numbers out loud once** before recording. The 0.507/0.846 pair is
  the spine of the whole thing and it is easy to transpose under pressure.

Recording: QuickTime (⌘⇧5) on macOS, OBS anywhere. Record system audio *and*
microphone if you narrate live; otherwise record silent and lay narration over
it. Export H.264 MP4, 1080p. Keep it under 3 minutes — most judging rubrics
stop watching there.

---

## Shot list

Times are cumulative. Each block is one continuous take; cut between blocks.

### 0:00 — The problem, and the thing everyone gets wrong (35 s)
*On screen: the app's landing view, or the title slide from `demo.mp4`.*

> US flight delay costs about $33 billion a year, and two thirds of that lands
> on passengers rather than airlines. A New York operations desk sees several
> hundred departures a day and can act on a few dozen — swap an aircraft, hold
> a connection. The constraint is attention, not information.
>
> Nearly every published model on this dataset reports an AUC above 0.90. They
> get there by using the observed departure delay — how late the plane actually
> left. By the time you know that, the decision is gone. So we measured the
> shortcut instead of taking it: same model, same split, **0.846 PR-AUC with it
> and 0.507 without**. Two thirds of the field's apparent skill is the
> observation that the plane left late.

### 0:35 — What we built (25 s)
*On screen: scroll the model card, or hold on the landing view.*

> LightGBM, 67 features all known before push-back, and a strictly temporal
> split — train on January to August, hold out November and December. PR-AUC
> 0.507 against a 0.250 base rate. The riskiest 10% of a day is 64% late.

### 1:00 — One flight (35 s)
*Tab **1 · One flight**. Pick a December flight with a visible risk score.
Point at the reasons panel.*

> A real flight from the held-out period. 61% risk. These reasons are the
> actual model — SHAP values, exact, not a narrative we wrote afterwards. Here
> it is the gust history and no slack in the aircraft's rotation. And the
> ground truth underneath: it landed 78 minutes late. The model never saw that.

### 1:35 — The desk view, and the honest number (40 s)
*Tab **2 · A whole day**. Move the alert-budget slider. Then tab **3 · What it
is worth**.*

> This is the operations desk. 22 December, 900 departures, a budget of 10%.
> Notice the budget is spent **per day** — a desk cannot save November's alerts
> for Christmas. That makes the number worse and the exercise real.
>
> And here is what it is worth. A 10% budget reaches **24% of all delay
> minutes**. Random alerting reaches 10%. A spreadsheet of historical route
> rates — which is what a desk actually has today — reaches 15%. So we are 2.4
> times random and 1.65 times the no-machine-learning alternative. At a
> deliberately pessimistic 10% recovery rate that is $3 million over two
> months, and $1.7 million of it is attributable to the model rather than to
> the budget existing at all.

### 2:15 — The finding we are proudest of (30 s)
*Tab **4 · Who it reaches**.*

> Then we asked who the alerts actually reach. Coverage of genuinely-late
> flights ranges from 38% down to 2% across carriers. Southwest runs late more
> often than JetBlue and gets a fortieth of the alerts, because the model
> under-predicts it by 17 points — and the AUC cannot see that at all.
> Spending the same budget proportionally closes the gap for 11.6% of the delay
> caught. For destination size the same change is actually free. We priced it
> instead of naming it.

### 2:45 — Close (20 s)
*Switch to a terminal. Run `make verify` if you have the artefacts — the
scrolling green is the shot. Otherwise hold on `reports/report.pdf`.*

> Two last things. We held out each airport in turn and retrained without it: a
> model deployed where it has no local history keeps 92% of its skill. And
> every number in that report is re-read from a metrics file by `make verify`
> — 119 checks — so the prose cannot drift away from the results. The whole
> pipeline rebuilds bit-identically from the raw CSVs, on two different CPU
> architectures.

---

## Rebuilding the generated video

```bash
make video                              # -> docs/demo.mp4
python scripts/build_demo_video.py --seconds 4   # faster cut
```

Needs an ffmpeg. `pip install imageio-ffmpeg` gives a self-contained one and is
what the script prefers; a system ffmpeg on `PATH` works too. If the slides
render but encoding aborts, the system ffmpeg is probably broken — install the
pip one and it will be picked up automatically.

The slide text lives in `build_slides()` in `scripts/build_demo_video.py`.
Numbers are pulled from `reports/metrics/` at render time on purpose: edit the
prose freely, but do not hard-code a figure, or the video becomes the one
artefact in this project that `make verify` cannot check.
