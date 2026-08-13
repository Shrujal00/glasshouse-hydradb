#!/usr/bin/env python
"""Load the resolved ontology into the HydraDB engine.

    python scripts/load_graph.py            # everything
    python scripts/load_graph.py --limit 5000

Writes the ontology as a graph rather than as a report of one:

    (:Alias {surface, kind, occurrences})
        -[:RESOLVES_TO {score, signals}]-> (:Entity {eid, canonical_name, confidence})

The evidence lives on the edge, which is the whole point. "Why do you think
`@mchen` is Maya Chen?" is then a traversal that answers itself, and the same
traversal is what the reasoning canvas animates.

Also writes a small SQLite lookup (surface -> entity) so that at question time
we can resolve the handful of names a question touches in microseconds without
loading a 58 MB ontology into memory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.config import STATE  # noqa: E402
from glasshouse.graph import GraphEngine, node_id  # noqa: E402

ENTITIES = STATE / "entities.jsonl"
LOOKUP = STATE / "ontology.sqlite3"

# The engine's admission control rejects batches over 1024 rows outright.
BATCH = 1000

LABELS = ("Entity", "Alias", "Document")


def open_lookup() -> sqlite3.Connection:
    conn = sqlite3.connect(LOOKUP)
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        DROP TABLE IF EXISTS alias;
        CREATE TABLE alias (
            surface TEXT, kind TEXT, eid TEXT, node_id INTEGER,
            canonical_name TEXT, confidence REAL, alias_count INTEGER
        );
        """
    )
    return conn


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run(limit: int | None, wipe: bool) -> None:
    if not ENTITIES.exists():
        raise SystemExit("no entities.jsonl; run scripts/resolve_entities.py first")

    engine = GraphEngine()
    if not engine.wait_until_ready(90):
        raise SystemExit("engine not reachable; is `docker compose up -d` running?")
    if wipe:
        # Deterministic node ids plus MERGE make loading idempotent, so a
        # rerun overwrites in place. Wiping is offered but not the default:
        # DETACH DELETE over a loaded ontology exceeds the 30s query ceiling.
        print("clearing previous ontology ...", flush=True)
        engine.wipe(LABELS)

    lookup = open_lookup()
    t0 = time.time()

    entity_rows: list[dict] = []
    alias_rows: list[dict] = []
    edge_rows: list[dict] = []
    lookup_rows: list[tuple] = []
    counts = {"entities": 0, "aliases": 0, "edges": 0}

    def flush() -> None:
        """Nodes before edges, because CREATE matches existing endpoints."""
        for batch in chunks(entity_rows, BATCH):
            engine.upsert_nodes(
                "Entity", batch, ["eid", "canonical_name", "confidence", "alias_count", "occurrences"]
            )
        for batch in chunks(alias_rows, BATCH):
            engine.upsert_nodes("Alias", batch, ["surface", "kind", "occurrences"])
        for batch in chunks(edge_rows, BATCH):
            engine.merge_edges(
                "RESOLVES_TO", batch, ["score", "signals"], src_label="Alias", dst_label="Entity"
            )
        lookup.executemany("INSERT INTO alias VALUES (?, ?, ?, ?, ?, ?, ?)", lookup_rows)
        lookup.commit()
        entity_rows.clear()
        alias_rows.clear()
        edge_rows.clear()
        lookup_rows.clear()

    with ENTITIES.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                break
            e = json.loads(line)
            eid_node = node_id(f"entity:{e['eid']}")
            entity_rows.append(
                {
                    "id": eid_node,
                    "eid": e["eid"],
                    "canonical_name": e["canonical_name"],
                    "confidence": float(e["confidence"]),
                    "alias_count": int(e["alias_count"]),
                    "occurrences": int(e["occurrences"]),
                }
            )
            counts["entities"] += 1

            # The evidence for each alias, keyed by the surface it justified.
            reason: dict[str, tuple[float, str]] = {}
            for m in e.get("merges") or ():
                sig = "+".join(m["signals"])
                for side in (m["a"], m["b"]):
                    prior = reason.get(side)
                    if prior is None or m["score"] > prior[0]:
                        reason[side] = (float(m["score"]), sig)

            for a in e["aliases"]:
                key = f"{a['kind']}:{a['surface']}"
                alias_node = node_id(f"alias:{key}")
                alias_rows.append(
                    {
                        "id": alias_node,
                        "surface": a["surface"],
                        "kind": a["kind"],
                        "occurrences": int(a["occurrences"]),
                    }
                )
                score, signals = reason.get(key, (1.0, "SOLE_SURFACE"))
                edge_rows.append(
                    {
                        "src": alias_node,
                        "dst": eid_node,
                        "score": score,
                        "signals": signals,
                    }
                )
                lookup_rows.append(
                    (
                        a["surface"],
                        a["kind"],
                        e["eid"],
                        eid_node,
                        e["canonical_name"],
                        float(e["confidence"]),
                        int(e["alias_count"]),
                    )
                )
                counts["aliases"] += 1
                counts["edges"] += 1

            if len(alias_rows) >= BATCH:
                flush()
                if counts["entities"] % 20_000 < 50:
                    rate = counts["entities"] / max(time.time() - t0, 1e-6)
                    print(
                        f"  {counts['entities']:>7,} entities  {rate:,.0f}/s", flush=True
                    )
    flush()

    print("  indexing lookup ...", flush=True)
    lookup.executescript(
        "CREATE INDEX alias_surface ON alias(surface);"
        "CREATE INDEX alias_eid ON alias(eid);"
    )
    lookup.commit()

    dt = time.time() - t0
    print(f"\n{'='*66}")
    print(f"  entities {counts['entities']:>9,}")
    print(f"  aliases  {counts['aliases']:>9,}")
    print(f"  edges    {counts['edges']:>9,}")
    print(f"  loaded in {dt:.1f}s ({counts['aliases']/max(dt,1e-6):,.0f} aliases/s)")
    print(f"  engine reports {engine.count('Entity'):,} Entity nodes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max entities to load")
    ap.add_argument("--wipe", action="store_true", help="clear existing nodes first (small graphs only)")
    args = ap.parse_args()
    run(args.limit, wipe=args.wipe)


if __name__ == "__main__":
    main()
