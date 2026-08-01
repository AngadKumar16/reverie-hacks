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

import logging
import sys
from typing import Dict

import pandas as pd

from src.config import DATA_RAW

log = logging.getLogger(__name__)

TABLES = ["flights", "weather", "planes", "airports", "airlines"]

# Expected row counts -- a cheap integrity check that we loaded the real thing.
EXPECTED_SHAPES = {
    "flights": (336776, 19),
    "weather": (26115, 15),
    "planes": (3322, 9),
    "airports": (1458, 8),
    "airlines": (16, 2),
}


def _from_package() -> Dict[str, pd.DataFrame]:
    try:
        import nycflights13 as nyc
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Neither CSVs in data/raw/ nor the `nycflights13` package were found.\n"
            "Fix with either:\n"
            "  pip install nycflights13\n"
            "or download the Kaggle dataset listed in the README into data/raw/."
        ) from exc
    return {name: getattr(nyc, name).copy() for name in TABLES}


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
