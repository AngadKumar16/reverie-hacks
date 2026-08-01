"""Build the modelling dataset once and cache it.

Running this module writes three parquet files plus a manifest describing the
feature contract, so that EDA, training, evaluation and the Streamlit app all
consume byte-identical inputs.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List, Tuple

import pandas as pd

from src import features as F
from src.config import DATA_PROCESSED, MODE_A, MODE_B, SEED
from src.data_loader import load_tables

log = logging.getLogger(__name__)

SPLIT_FILES = {
    "train": DATA_PROCESSED / "train.parquet",
    "valid": DATA_PROCESSED / "valid.parquet",
    "test": DATA_PROCESSED / "test.parquet",
}
MANIFEST = DATA_PROCESSED / "manifest.json"
FULL_FILE = DATA_PROCESSED / "flights_features.parquet"


def build(force: bool = False) -> Dict[str, object]:
    if MANIFEST.exists() and not force:
        log.info("processed data already built; use --force to rebuild")
        return json.loads(MANIFEST.read_text())

    tables = load_tables()
    df = F.build_feature_frame(tables)
    df.to_parquet(FULL_FILE, index=False)

    train, valid, test = F.temporal_split(df)
    train, (valid, test), encodings = F.add_target_encodings(train, [valid, test])

    for name, frame in zip(SPLIT_FILES, (train, valid, test)):
        frame.to_parquet(SPLIT_FILES[name], index=False)

    manifest = {
        "seed": SEED,
        "n_rows_total": int(len(df)),
        "rows": {k: int(len(v)) for k, v in
                 zip(SPLIT_FILES, (train, valid, test))},
        "late_rate": {k: float(v["is_delayed"].mean()) for k, v in
                      zip(SPLIT_FILES, (train, valid, test))},
        "encodings": encodings,
        "features": {
            MODE_A: F.feature_columns(train, MODE_A, encodings),
            MODE_B: F.feature_columns(train, MODE_B, encodings),
        },
        "categorical": F.CATEGORICAL_FEATURES,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    log.info("wrote processed splits to %s", DATA_PROCESSED)
    return manifest


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    if not MANIFEST.exists():
        build()
    manifest = json.loads(MANIFEST.read_text())
    frames = [pd.read_parquet(SPLIT_FILES[k]) for k in ("train", "valid", "test")]
    return frames[0], frames[1], frames[2], manifest


def load_full() -> pd.DataFrame:
    if not FULL_FILE.exists():
        build()
    return pd.read_parquet(FULL_FILE)


def xy(df: pd.DataFrame, cols: List[str]):
    return F.as_model_frame(df, cols), df["is_delayed"].to_numpy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    m = build(force="--force" in sys.argv)
    print(json.dumps({k: v for k, v in m.items() if k != "features"}, indent=2))
    print("n features (preflight):", len(m["features"][MODE_A]))
    print("n features (gate):     ", len(m["features"][MODE_B]))
