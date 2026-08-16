"""Load the NYC Flights 2013 tables.

The project supports two interchangeable sources so that the repository can be
reproduced with or without a Kaggle account:

1. **CSV files in ``data/raw/``** -- download the Kaggle dataset (see README)
   and drop ``flights.csv``, ``weather.csv``, ``planes.csv``, ``airports.csv``
   and ``airlines.csv`` there. This path is used automatically if the files
   exist.
2. **The ``nycflights13`` PyPI package** -- ships the identical tables from the
   original tidyverse R data package. Used as a fallback and to bootstrap the
   CSVs on a fresh clone.

Both sources trace back to the same primary data: the US Bureau of
Transportation Statistics on-time performance records for 2013, joined with the
FAA aircraft registry and ASOS/NOAA hourly weather observations.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import DATA_RAW

log = logging.getLogger(__name__)

TABLES = ["flights", "weather", "planes", "airports", "airlines"]

# The filenames the `nycflights13` wheel ships in its ``data/`` directory.
# ``flights`` is zipped; the other four are plain CSVs.
_PACKAGE_FILES = {
    "flights": "flights.csv.zip",
    "weather": "weather.csv",
    "planes": "planes.csv",
    "airports": "airports.csv",
    "airlines": "airlines.csv",
}

# Expected row counts -- a cheap integrity check that we loaded the real thing.
EXPECTED_SHAPES = {
    "flights": (336776, 19),
    "weather": (26115, 15),
    "planes": (3322, 9),
    "airports": (1458, 8),
    "airlines": (16, 2),
}


def _package_data_dir() -> Path | None:
    """Locate the ``data/`` directory inside an installed ``nycflights13``.

    Deliberately does *not* ``import nycflights13``. That package's ``__init__``
    imports ``pkg_resources``, which is absent from a fresh Python 3.12+
    virtualenv -- setuptools stopped being installed by default, and
    ``pkg_resources`` is deprecated under PEP 632. Importing it therefore fails
    with ``ModuleNotFoundError: pkg_resources`` on a current Python even though
    the package and its data are sitting right there on disk.

    ``find_spec`` resolves the install location without executing the module,
    and the five tables are plain files in it, so reading them directly makes
    the no-Kaggle-account path work on every supported Python.
    """
    spec = importlib.util.find_spec("nycflights13")
    if spec is None or not spec.origin:
        return None
    data_dir = Path(spec.origin).parent / "data"
    return data_dir if data_dir.is_dir() else None


def _from_package() -> Dict[str, pd.DataFrame]:
    data_dir = _package_data_dir()
    if data_dir is None:
        raise SystemExit(
            "Neither CSVs in data/raw/ nor the `nycflights13` package were found.\n"
            "Fix with either:\n"
            "  pip install -r requirements.txt\n"
            "or download the Kaggle dataset listed in the README into data/raw/."
        )
    missing = [f for f in _PACKAGE_FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"`nycflights13` is installed at {data_dir} but is missing "
            f"{', '.join(missing)}.\nReinstall it, or download the Kaggle "
            "dataset listed in the README into data/raw/."
        )
    return {name: pd.read_csv(data_dir / fname)
            for name, fname in _PACKAGE_FILES.items()}


def _from_csv() -> Dict[str, pd.DataFrame]:
    return {name: pd.read_csv(DATA_RAW / f"{name}.csv") for name in TABLES}


def csvs_present() -> bool:
    return all((DATA_RAW / f"{name}.csv").exists() for name in TABLES)


def materialise_csvs(overwrite: bool = False) -> None:
    """Write the five tables to ``data/raw/`` so the repo is self-contained."""
    if csvs_present() and not overwrite:
        log.info("Raw CSVs already present; nothing to do.")
        return
    tables = _from_package()
    for name, df in tables.items():
        out = DATA_RAW / f"{name}.csv"
        df.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out.name, len(df))


def load_tables(verify: bool = True) -> Dict[str, pd.DataFrame]:
    """Return the five raw tables as a dict of DataFrames."""
    tables = _from_csv() if csvs_present() else _from_package()

    if verify:
        for name, expected in EXPECTED_SHAPES.items():
            got = tables[name].shape
            if got != expected:
                log.warning(
                    "%s has shape %s, expected %s -- continuing, but the "
                    "numbers in the report assume the canonical dataset.",
                    name, got, expected,
                )
    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    materialise_csvs(overwrite="--force" in sys.argv)
    tabs = load_tables()
    for k, v in tabs.items():
        print(f"{k:10s} {v.shape[0]:>7,} rows x {v.shape[1]:>2} cols")
