"""Diagnose why the app or the pipeline will not start.

Deliberately uses **only the standard library**, so it runs on an interpreter
that has none of the project's dependencies installed — which is exactly the
situation it exists to diagnose.

    python3 scripts/doctor.py

Reports the interpreter in use, which required packages are missing from *that*
interpreter, and whether the generated data and models are on disk. Prints the
exact command to fix whatever it finds.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# import name -> pip name (they differ for scikit-learn)
REQUIRED = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("shap", "shap"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("joblib", "joblib"),
    ("pyarrow", "pyarrow"),
    ("streamlit", "streamlit"),
    ("nycflights13", "nycflights13"),
]

APP_REQUIRED = ["streamlit", "lightgbm", "shap", "joblib", "pandas", "numpy",
                "matplotlib", "pyarrow"]

DATA_FILES = [
    ("data/processed/manifest.json", "make data"),
    ("data/processed/train.parquet", "make data"),
    ("data/processed/valid.parquet", "make data"),
    ("data/processed/test.parquet", "make data"),
]
MODEL_FILES = [
    ("models/lightgbm.joblib", "make train"),
    ("models/lightgbm_severity.joblib", "make train"),
    ("models/training_context.joblib", "make train"),
]

OK, BAD, WARN = "  ok  ", " MISS ", " warn "


def line(status: str, text: str, detail: str = "") -> None:
    print(f"[{status}] {text}" + (f"  — {detail}" if detail else ""))


def version_of(mod: str) -> str:
    try:
        import importlib.metadata as md
        name = {"sklearn": "scikit-learn"}.get(mod, mod)
        return md.version(name)
    except Exception:
        return "?"


def main() -> int:
    print("\n=== interpreter ===")
    print(f"  python      {sys.version.split()[0]}  ({platform.machine()})")
    print(f"  executable  {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    venv_dir = ROOT / ".venv"
    line(OK if in_venv else WARN,
         "running inside a virtual environment" if in_venv
         else "NOT running inside a virtual environment",
         "" if in_venv else "this is the usual cause of a split install")
    if venv_dir.exists() and not in_venv:
        line(WARN, f".venv exists at {venv_dir} but is not active",
             "run: source .venv/bin/activate")

    print("\n=== packages (in THIS interpreter) ===")
    missing = []
    for mod, pip_name in REQUIRED:
        found = importlib.util.find_spec(mod) is not None
        if found:
            line(OK, f"{mod:<14}", version_of(mod))
        else:
            line(BAD, f"{mod:<14}", f"pip install {pip_name}")
            missing.append(pip_name)

    app_missing = [m for m, _ in REQUIRED
                   if m in APP_REQUIRED and importlib.util.find_spec(m) is None]

    print("\n=== generated files ===")
    stale = set()
    for rel, fix in DATA_FILES + MODEL_FILES:
        path = ROOT / rel
        if path.exists():
            size = path.stat().st_size
            line(OK, rel, f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size} B")
        else:
            line(BAD, rel, f"run: {fix}")
            stale.add(fix)

    figs = list((ROOT / "reports" / "figures").glob("*.png"))
    line(OK if len(figs) == 26 else WARN, f"reports/figures  {len(figs)}/26 png",
         "" if len(figs) == 26 else "run: make eda evaluate explain severity horizon disruption")

    # --------------------------------------------------------------
    print("\n=== verdict ===")
    if missing:
        print(f"  {len(missing)} package(s) missing from {sys.executable}")
        print("\n  Fix — create an isolated environment and install everything into it:\n")
        print("    cd " + str(ROOT))
        print("    python3 -m venv .venv")
        print("    source .venv/bin/activate")
        print("    pip install -r requirements.txt")
        print("\n  Then re-run this check:  python3 scripts/doctor.py")
        if app_missing:
            print(f"\n  The app specifically needs: {', '.join(app_missing)}")
        return 1

    if stale:
        print("  Dependencies are fine, but generated files are missing.")
        print("  Fix:\n")
        for fix in sorted(stale):
            print(f"    {fix}")
        return 1

    print("  Everything the app needs is present. Start it with:\n")
    print("    make app")
    print("  or")
    print(f"    {sys.executable} -m streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
