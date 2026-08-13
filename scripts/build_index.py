#!/usr/bin/env python
"""Build the local full-text index over the normalized corpus.

    python scripts/build_index.py

Takes a few minutes over half a million documents and needs no network, no
account and no API key - which is the point. A judge clones the repo, runs
this, and can ask questions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.corpus import SOURCES  # noqa: E402
from glasshouse.recall import LocalRecall, iter_normalized  # noqa: E402

BATCH = 5_000


def run(sources: list[str]) -> None:
    recall = LocalRecall()
    recall.create()
    t0 = time.time()
    batch: list[tuple] = []
    total = 0

    for row in iter_normalized(sources):
        batch.append(row)
        if len(batch) >= BATCH:
            recall.add(batch)
            total += len(batch)
            batch.clear()
            if total % 50_000 == 0:
                rate = total / max(time.time() - t0, 1e-6)
                print(f"  {total:>7,} docs  {rate:,.0f}/s", flush=True)
    if batch:
        recall.add(batch)
        total += len(batch)

    recall.conn.commit()
    print(f"  indexed {total:,} in {time.time()-t0:.1f}s, optimizing ...", flush=True)
    recall.optimize()

    size = recall.path.stat().st_size / 1e9
    print(f"\nindex: {recall.count():,} documents, {size:.2f} GB at {recall.path}")
    print(f"built in {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", help="restrict to source(s)")
    args = ap.parse_args()
    run(args.source or list(SOURCES))


if __name__ == "__main__":
    main()
