#!/usr/bin/env python
"""Load the lookup half of the ontology: written forms, and who they denote.

    python scripts/load_surface_graph.py
    python scripts/load_surface_graph.py --limit 5000 --wipe

`load_graph.py` already puts `(:Alias)-[:RESOLVES_TO]->(:Entity)` in HydraDB,
but nothing reads it: resolution at query time went to a SQLite mirror instead,
because the Alias node is keyed on `kind:surface` and a question does not tell
you the kind. A question contains the string `sam` and nothing else.

So this adds the entrance that a question can actually use:

    (:Surface {text, kinds, entities, given_name_forms})-[:DENOTES]->(:Entity)

keyed on the written form alone, so `node_id(f"surface:{text}")` turns any word
in a question into an anchored lookup. One hop out reaches every person that
form could mean, and *how many it reaches is the ambiguity guard* -- `sam` is
eight people in this corpus, and a resolution path that returns eight is a
question that has not named anybody.

Two corpus-wide facts are precomputed onto the Surface node, because the engine
answers anchored traversals and rejects scans:

  `entities`          how many people this form denotes. The ambiguity guard.
  `kinds`             every role the form was written in, pipe-joined. One
                      spelling used as both a handle and a name is what makes a
                      person known only by a first name admissible.
  `given_name_forms`  how many distinct full names start with this word, which
                      is what separates a capitalised "Jordan" that is somebody
                      from a capitalised "Runtime" that is not.

Both are counted here, once, over the whole ontology. Query time then never
needs to look at anything but the node it anchored on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.config import STATE  # noqa: E402
from glasshouse.graph import GraphEngine, GraphError, node_id  # noqa: E402

ENTITIES = STATE / "entities.jsonl"
BATCH = 1000
WRITE_RETRIES = 3
LABELS = ("Surface",)


def scan(limit: int | None) -> tuple[dict[str, dict], list[dict], Counter]:
    """One pass over the ontology, collecting surfaces and their entities.

    Held in memory deliberately: `entities` and `given_name_forms` are counts
    over the whole file, so nothing can be written until the whole file has
    been read. 209,388 surfaces is a few tens of megabytes.
    """
    surfaces: dict[str, dict] = {}
    denotes: list[dict] = []
    given: Counter = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    kinds_seen: dict[str, set[str]] = defaultdict(set)
    counts: Counter = Counter()

    with ENTITIES.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh):
            if limit is not None and line_number >= limit:
                break
            entity = json.loads(line)
            entity_node = node_id(f"entity:{entity['eid']}")
            counts["entities"] += 1

            for alias in entity.get("aliases") or ():
                text = str(alias.get("surface") or "").strip().lower()
                if not text:
                    continue
                kind = str(alias.get("kind") or "")
                record = surfaces.get(text)
                if record is None:
                    record = surfaces[text] = {
                        "id": node_id(f"surface:{text}"),
                        "text": text,
                        # Every role the form was written in, pipe-joined. One
                        # spelling used both as a handle and as a name is the
                        # corroboration that separates a person known by a
                        # first name from a channel tag, so collapsing the
                        # forms by text must not collapse their kinds too.
                        "kinds": "",
                        "entities": 0,
                        "given_name_forms": 0,
                    }
                    counts["surfaces"] += 1
                kinds_seen[text].add(kind)
                # The same person can hold one written form once only; two
                # entities holding it is exactly what `entities` must count.
                if (text, entity["eid"]) in seen_pairs:
                    continue
                seen_pairs.add((text, entity["eid"]))
                record["entities"] += 1
                denotes.append(
                    {
                        "id": node_id(f"denotes:{text}:{entity['eid']}"),
                        "src": record["id"],
                        "dst": entity_node,
                        "kind": kind,
                        "occurrences": int(alias.get("occurrences") or 0),
                    }
                )
                counts["denotes"] += 1

                # "jordan reyes" contributes evidence that "jordan" is a given
                # name. Only multi-word personal names count: an email address
                # or a handle says nothing about what a bare word means.
                if kind == "name" and " " in text:
                    given[text.split(" ", 1)[0]] += 1

    for text, kinds in kinds_seen.items():
        surfaces[text]["kinds"] = "|".join(sorted(k for k in kinds if k))
    for word, n in given.items():
        record = surfaces.get(word)
        if record is not None:
            record["given_name_forms"] = int(n)
    return surfaces, denotes, counts


def write(engine: GraphEngine, surfaces: dict[str, dict], denotes: list[dict]) -> None:
    rows = list(surfaces.values())

    def flush(fn, items, what):
        for start in range(0, len(items), BATCH):
            chunk = items[start : start + BATCH]
            for attempt in range(WRITE_RETRIES):
                try:
                    fn(chunk)
                    break
                except GraphError:
                    if attempt == WRITE_RETRIES - 1:
                        raise
                    time.sleep(0.4 * (attempt + 1))
            done = min(start + BATCH, len(items))
            if done % 20_000 == 0 or done == len(items):
                print(f"  {what:<9} {done:>7,}/{len(items):,}", flush=True)

    # Nodes before edges: the edge MERGE matches existing endpoints and will
    # silently write nothing for an endpoint that does not exist yet.
    flush(lambda c: engine.upsert_nodes(
        "Surface", c, ["text", "kinds", "entities", "given_name_forms"]), rows, "surfaces")
    flush(lambda c: engine.merge_edges(
        "DENOTES", c, ["kind", "occurrences"],
        src_label="Surface", dst_label="Entity"), denotes, "denotes")


def run(limit: int | None, wipe: bool) -> None:
    if not ENTITIES.exists():
        raise SystemExit("no entities.jsonl; run scripts/resolve_entities.py first")
    engine = GraphEngine()
    if not engine.wait_until_ready(90):
        raise SystemExit("engine not reachable; is `docker compose up -d` running?")
    if wipe:
        print("clearing previous surfaces ...", flush=True)
        engine.wipe(LABELS)

    t0 = time.time()
    print("scanning the ontology ...", flush=True)
    surfaces, denotes, counts = scan(limit)
    ambiguous = sum(1 for s in surfaces.values() if s["entities"] > 1)
    print(f"  {counts['entities']:,} entities, {counts['surfaces']:,} surfaces, "
          f"{counts['denotes']:,} denotes edges, {ambiguous:,} ambiguous forms",
          flush=True)

    write(engine, surfaces, denotes)
    elapsed = time.time() - t0
    written = len(surfaces) + len(denotes)
    print(f"\nloaded {written:,} items in {elapsed:.1f}s "
          f"({written/max(elapsed,1e-6)*60:,.0f}/min)")
    # The forms a question is most likely to contain and least able to resolve.
    worst = sorted(surfaces.values(), key=lambda s: -s["entities"])[:5]
    for s in worst:
        print(f"  most ambiguous  {s['text']:<28} denotes {s['entities']} people")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="entities to read")
    ap.add_argument("--wipe", action="store_true", help="clear Surface nodes first")
    args = ap.parse_args()
    run(args.limit, args.wipe)


if __name__ == "__main__":
    main()
