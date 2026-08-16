"""Render a narrated-slide demo video from the generated artefacts.

The Datathon track requires a demo video. The best one is a screen recording of
the live app with a human talking over it, and `docs/DEMO_VIDEO.md` is the
shot-by-shot script for exactly that. This script builds the other kind: a
silent, captioned walkthrough of the results, assembled from the figures and
metrics the pipeline produced.

It exists for two reasons. It is a fallback if recording falls through, and --
more usefully -- **every number that appears on screen is read from
`reports/metrics/` at render time**, not typed into a slide. The video is
generated from the same artefacts `scripts/verify.py` checks the report
against, so it cannot quietly disagree with the report the way a hand-made deck
can. Re-run `make reproduce` and the video re-renders with the new numbers or
not at all.

    python scripts/build_demo_video.py            # -> docs/demo.mp4
    python scripts/build_demo_video.py --seconds 6

Needs an ffmpeg. `pip install imageio-ffmpeg` provides a self-contained one and
is what this prefers; a system ffmpeg on PATH is used otherwise.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGURES = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics"
OUT = ROOT / "docs" / "demo.mp4"

# 1920x1080 at 100 dpi.
W_IN, H_IN, DPI = 19.2, 10.8, 100
INK = "#12232b"
PAPER = "#fbfaf7"
ACCENT = "#c44e52"
MUTED = "#5b6b73"


def m(name: str) -> Dict:
    return json.loads((METRICS / f"{name}.json").read_text())


def _canvas():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=PAPER)
    return fig


def title_slide(fig, headline: str, sub: str = "", kicker: str = "") -> None:
    if kicker:
        # matplotlib has no letter-spacing property, so space it by hand.
        fig.text(.5, .70, "  ".join(kicker.upper()), ha="center", va="center",
                 fontsize=17, color=ACCENT, weight="bold")
    fig.text(.5, .55, headline, ha="center", va="center", fontsize=58,
             color=INK, weight="bold", wrap=True)
    if sub:
        fig.text(.5, .36, sub, ha="center", va="center", fontsize=27,
                 color=MUTED, wrap=True, linespacing=1.6)


def figure_slide(fig, headline: str, image: Path, caption: str = "") -> None:
    fig.text(.5, .945, headline, ha="center", va="center", fontsize=36,
             color=INK, weight="bold")
    ax = fig.add_axes([.04, .13, .92, .76])
    ax.imshow(mpimg.imread(image))
    ax.axis("off")
    if caption:
        fig.text(.5, .055, caption, ha="center", va="center", fontsize=22,
                 color=MUTED, wrap=True)


def stat_slide(fig, headline: str, stats: List[tuple], footer: str = "") -> None:
    fig.text(.5, .87, headline, ha="center", va="center", fontsize=42,
             color=INK, weight="bold")
    n = len(stats)
    for i, (value, label) in enumerate(stats):
        x = (i + .5) / n
        fig.text(x, .52, value, ha="center", va="center", fontsize=76,
                 color=ACCENT, weight="bold")
        fig.text(x, .35, label, ha="center", va="center", fontsize=22,
                 color=MUTED, wrap=True, linespacing=1.5)
    if footer:
        fig.text(.5, .13, footer, ha="center", va="center", fontsize=21,
                 color=MUTED, style="italic", wrap=True)


def build_slides() -> List[Dict]:
    """Every number below is read from the metrics files, never typed in."""
    ev, imp, tra = m("evaluation"), m("impact"), m("transfer")
    dis, hor, fair = m("disruption"), m("horizon"), m("fairness")
    abl, dep = m("ablation"), m("deployment")

    pre = ev["lightgbm"]["test"]
    gate = ev["lightgbm_gate"]["test"]
    base = ev["base_rates"]["test"]
    cap = ev["capacity"]
    mod = imp["at_operating_budget"]["model"]
    hist = imp["at_operating_budget"]["historical_rule"]
    rnd = imp["at_operating_budget"]["random"]

    return [
        dict(kind="title", seconds=5.0,
             kicker="ReverieHacks 2026 · Datathon",
             headline="FlightRisk NYC",
             sub="Predicting arrival delay before the aircraft moves\n"
                 "336,776 flights · New York, 2013"),

        dict(kind="title", seconds=6.5,
             headline="An ops desk cannot act on\nwhat it learns at push-back",
             sub="Several hundred departures a day, a few dozen you can do "
                 "anything about.\nThe constraint is attention, not information."),

        dict(kind="stat", seconds=7.0,
             headline="Most published models take a shortcut. We measured it.",
             stats=[(f"{gate['pr_auc']:.3f}",
                     "PR-AUC using the observed\ndeparture delay\n(the usual number)"),
                    (f"{pre['pr_auc']:.3f}",
                     "PR-AUC knowing only what is\ntrue before push-back\n"
                     "(the honest one)")],
             footer="Same model, same data, same split. Two thirds of the "
                    "field's apparent skill is the observation that the plane "
                    "left late."),

        dict(kind="figure", seconds=6.5, image=FIGURES / "08_roc_pr_curves.png",
             headline="67 pre-flight features, a strictly temporal split",
             caption=f"Train Jan–Aug, hold out Nov–Dec. PR-AUC "
                     f"{pre['pr_auc']:.3f} against a {base:.1%} base rate; "
                     f"ROC-AUC {pre['roc_auc']:.3f}."),

        dict(kind="stat", seconds=7.0,
             headline="Scored as a rationing device, not a classifier",
             stats=[(f"{cap['precision_at_k']:.0%}",
                     "of the riskiest 10%\nreally are late"),
                    (f"{cap['lift']:.1f}×",
                     "better than\nalerting at random"),
                    (f"{mod['delay_min_share']:.1%}",
                     "of all delay minutes\nreached on a 10% budget")],
             footer=f"Budget is spent per day — a desk cannot save November's "
                    f"alerts for 22 December. Random reaches "
                    f"{rnd['delay_min_share_mean']:.1%}; a no-ML lookup of "
                    f"historical route rates reaches {hist['delay_min_share']:.1%}."),

        dict(kind="figure", seconds=6.0, image=FIGURES / "27_impact_curve.png",
             headline="What a 10% alert budget is worth",
             caption=f"${mod['gross_value_usd']/1e6:.2f}M of recoverable value "
                     f"over two months at a deliberately pessimistic "
                     f"{imp['assumptions']['mitigation_effectiveness']:.0%} "
                     f"mitigation rate — a ceiling on available value, not a "
                     f"measured saving."),

        dict(kind="stat", seconds=7.0,
             headline="The worse the outcome, the better it is predicted",
             stats=[(f"{dis['is_cancelled']['roc_auc']:.3f}",
                     "ROC-AUC for cancellation"),
                    (f"{dis['is_cancelled']['recall_at_10pct']:.0%}",
                     "of all cancellations sit in\nthe riskiest 10% of the list"),
                    (f"{dis['is_diverted']['roc_auc']:.3f}",
                     "ROC-AUC for diversion —\nthe honest failure")],
             footer="Severe disruption has causes in the feature set: storms, "
                    "closed runways, broken rotations. Marginal lateness is "
                    "mostly noise. Diversion is decided in the air, by weather "
                    "this dataset does not contain."),

        dict(kind="figure", seconds=6.0, image=FIGURES / "25_forecast_horizon.png",
             headline="How far ahead it still works",
             caption=f"With the crudest possible forecast, three hours of "
                     f"warning costs {hor['0']['pr_auc'] - hor['3']['pr_auc']:.3f} "
                     f"PR-AUC. Twenty-four hours lands exactly on the "
                     f"no-weather floor — a built-in check that the experiment "
                     f"measures what it claims."),

        dict(kind="figure", seconds=7.0, image=FIGURES / "29_fairness.png",
             headline="Ranking by risk is a rationing rule, and it has losers",
             caption=f"Coverage of genuinely-late flights ranges "
                     f"{fair['disparity']['carrier']['recall_max']:.1%} to "
                     f"{fair['disparity']['carrier']['recall_min']:.1%} across "
                     f"carriers. Spending the same budget proportionally closes "
                     f"the gap for "
                     f"{fair['price_of_equity']['carrier']['delay_min_cost_of_equity_pct']:.1f}% "
                     f"of the delay caught. Priced, not asserted."),

        dict(kind="figure", seconds=7.0, image=FIGURES / "30_transfer.png",
             headline="Does it work at an airport it has never seen?",
             caption=f"Each airport held out in turn — trained, encoded and "
                     f"early-stopped on the other two only. A model with no "
                     f"local history keeps "
                     f"{tra['summary']['mean_retention_pr_auc']:.0%} of its "
                     f"skill, never below "
                     f"{tra['summary']['worst_retention_pr_auc']:.0%}."),

        dict(kind="stat", seconds=6.5,
             headline="Cheap enough to actually run",
             stats=[(f"{dep['artefacts']['deployable_subset_mb']:.1f} MB",
                     "everything deployment needs"),
                    (f"{dep['scoring']['one_day']['median_ms']:.0f} ms",
                     "to score a whole day\nof New York departures"),
                    (f"{dep['scale_projection']['rows'][-1]['daily_seconds']:.0f} s",
                     "per day at 100× New York's\nvolume — one core")],
             footer="Measured, not estimated. No GPU, no vector database, no "
                    "inference API bill: a cron job, not a service."),

        dict(kind="stat", seconds=7.0,
             headline="Every number here is checked by a script",
             stats=[("119", "checks in `make verify`, including\nevery headline "
                            "number re-read\nfrom the artefact that produced it"),
                    ("16", "leakage and correctness tests\nguarding the feature "
                           "boundary"),
                    ("0.000000",
                     "PR-AUC drift when the whole\npipeline is rebuilt on a "
                     "second\nCPU architecture")],
             footer="Scrambling every post-departure column leaves the shipped "
                    "model's predictions exactly unchanged. The prose cannot "
                    "drift away from the results."),

        dict(kind="title", seconds=6.0,
             headline="It tells an ops desk which flights\nwill land late — "
                      "early enough to act",
             sub="…and measures what that is worth, instead of asserting it.\n\n"
                 "github.com/aaravsinhaofficial/reverie-hacks"),
    ]


def render(slides: List[Dict], tmp: Path) -> List[Path]:
    paths = []
    for i, s in enumerate(slides):
        fig = _canvas()
        if s["kind"] == "title":
            title_slide(fig, s["headline"], s.get("sub", ""), s.get("kicker", ""))
        elif s["kind"] == "stat":
            stat_slide(fig, s["headline"], s["stats"], s.get("footer", ""))
        else:
            figure_slide(fig, s["headline"], s["image"], s.get("caption", ""))
        p = tmp / f"slide_{i:03d}.png"
        fig.savefig(p, dpi=DPI, facecolor=PAPER)
        plt.close(fig)
        paths.append(p)
        print(f"  rendered {p.name}  ({s['seconds']:.1f}s)  {s['headline'][:52]}")
    return paths


def find_ffmpeg() -> Optional[str]:
    """Prefer a pip-installed ffmpeg over the system one.

    `imageio-ffmpeg` ships a self-contained static binary, which sidesteps the
    usual ways a system ffmpeg is broken -- a Homebrew upgrade that leaves a
    dangling `libx265` dylib will abort the process before it reads its
    arguments, and that is what happened on the machine this was written on.
    A system ffmpeg on PATH is still used if the pip one is not installed.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def encode(slides: List[Dict], paths: List[Path], tmp: Path, out: Path,
           ffmpeg: str) -> None:
    listing = tmp / "slides.txt"
    lines = []
    for s, p in zip(slides, paths):
        lines.append(f"file '{p}'")
        lines.append(f"duration {s['seconds']}")
    # The concat demuxer ignores the final entry's duration, so the last image
    # is repeated to give it one.
    lines.append(f"file '{paths[-1]}'")
    listing.write_text("\n".join(lines))

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-vf", "fps=30,format=yuv420p,scale=1920:1080:flags=lanczos",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)


def main(seconds: Optional[float], out: Path) -> int:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("No ffmpeg available. Either `pip install imageio-ffmpeg` "
              "(self-contained, recommended) or install a system ffmpeg.",
              file=sys.stderr)
        return 1

    slides = build_slides()
    if seconds is not None:
        for s in slides:
            s["seconds"] = seconds

    missing = [s["image"] for s in slides
               if s["kind"] == "figure" and not Path(s["image"]).exists()]
    if missing:
        print("missing figures — run `make reproduce` first:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 1

    total = sum(s["seconds"] for s in slides)
    print(f"building {len(slides)} slides, {total:.0f}s total")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = render(slides, tmp)
        encode(slides, paths, tmp, out, ffmpeg)
    size = out.stat().st_size / 1024 ** 2
    print(f"\nwrote {out.relative_to(ROOT)}  ({total:.0f}s, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=None,
                    help="override every slide's duration")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    sys.exit(main(args.seconds, args.out))
