"""Locate exactly which feature column breaks the determinism check.

`scripts/verify.py` hashes every numeric feature into one digest, so a mismatch
tells you a split drifted but not where. This script does the same rebuild and
then compares cached vs freshly-rebuilt splits column by column, reporting the
number of differing rows, the largest absolute delta, and a few example rows.

    .venv/bin/python scripts/diff_splits.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features as F                     # noqa: E402
from src.config import MODE_A                     # noqa: E402
from src.data_loader import load_tables           # noqa: E402
from src.pipeline import SPLIT_FILES, MANIFEST    # noqa: E402

NA = -9999.0
N_EXAMPLES = 5


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    feats = manifest["features"][MODE_A]

    tables = load_tables()
    rebuilt = F.build_feature_frame(tables)
    fresh = dict(zip(("train", "valid", "test"), F.temporal_split(rebuilt)))
    fresh["train"], (fresh["valid"], fresh["test"]), _ = F.add_target_encodings(
        fresh["train"], [fresh["valid"], fresh["test"]]
    )

    n_bad = 0
    for name in ("train", "valid", "test"):
        cached = pd.read_parquet(SPLIT_FILES[name])
        new = fresh[name]
        print(f"\n=== {name}: cached {len(cached):,} rows, rebuilt {len(new):,} rows ===")
        if len(cached) != len(new):
            print("   row counts differ — everything below is meaningless")
            n_bad += 1
            continue

        for col in feats:
            if col not in cached.columns:
                print(f"   {col}: absent from the cached parquet")
                n_bad += 1
                continue

            a, b = cached[col], new[col]
            if str(a.dtype) != str(b.dtype):
                print(f"   {col}: dtype cached={a.dtype} rebuilt={b.dtype}")

            if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
                x = np.nan_to_num(a.to_numpy(dtype="float64"), nan=NA)
                y = np.nan_to_num(b.to_numpy(dtype="float64"), nan=NA)
                mask = x != y
                if not mask.any():
                    continue
                idx = np.flatnonzero(mask)
                print(f"   {col}: {mask.sum():,} rows differ, "
                      f"max |delta| {np.abs(x[mask] - y[mask]).max():.6g}")
                for i in idx[:N_EXAMPLES]:
                    print(f"      row {i}: cached={x[i]!r} rebuilt={y[i]!r}")
                n_bad += 1
            else:
                mask = (a.astype(str).to_numpy() != b.astype(str).to_numpy())
                if not mask.any():
                    continue
                # verify.py ignores non-numeric columns, so flag these as FYI.
                print(f"   {col}: {mask.sum():,} rows differ (non-numeric, "
                      f"not part of the verify digest)")

    print("\nColumns above are the ones verify.py is hashing." if n_bad else
          "\nEvery numeric feature is identical — the digest should match.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
