#!/usr/bin/env python
"""Build the facet store: containers, headers and participants per document.

    python scripts/build_facets.py
    python scripts/build_facets.py --source confluence --limit 2000

Reads the same normalized shards `build_index.py` does, but keeps the fields
the full-text index throws away -- the folder, the channel, the speakers, the
mail headers -- which is where the answers to the metadata questions live. One
source at a time and restartable, so a single shard can be rebuilt without
touching the other eight.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.corpus import SOURCES  # noqa: E402
from glasshouse.facets import FacetStore  # noqa: E402


def run(sources: list[str], limit: int | None) -> None:
    store = FacetStore()
    t0 = time.time()
    total = 0

    for source in sources:
        start = time.time()
        counted = store.build([source], limit=limit)
        seen = counted.get(source, 0)
        total += seen
        rate = seen / max(time.time() - start, 1e-6)
        print(f"  {source:<13} {seen:>7,} docs  {time.time()-start:6.1f}s  {rate:,.0f}/s",
              flush=True)

    documents, containers = store.counts()
    size = store.path.stat().st_size / 1e9
    print(f"\nfacets: {documents:,} documents, {containers:,} containers, "
          f"{size:.2f} GB at {store.path}")
    # The largest containers are the ones a careless scope would open, so print
    # them: they are what `containers_named`'s two-token floor exists to hold shut.
    for container in store.all_containers(limit=5):
        print(f"  largest  {container.key:<50} {container.documents:>7,}")
    print(f"built {total:,} documents in {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", help="restrict to source(s)")
    ap.add_argument("--limit", type=int, help="stop after N documents per source")
    args = ap.parse_args()
    run(args.source or list(SOURCES), args.limit)


if __name__ == "__main__":
    main()
