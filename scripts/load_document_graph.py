#!/usr/bin/env python
"""Load exact document-to-entity links into HydraDB before question time.

    python scripts/load_document_graph.py --source slack --limit 100

Only normalized identity fields participate in linking. Document bodies are
never written to HydraDB or scanned for aliases.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.config import NORMALIZED, STATE  # noqa: E402
from glasshouse.graph import GraphEngine, GraphError, node_id  # noqa: E402
from glasshouse.priors import Priors  # noqa: E402
from glasshouse.resolve import norm_name  # noqa: E402

LOOKUP = STATE / "ontology.sqlite3"
CHECKPOINT = STATE / "document_graph_checkpoint.json"
BATCH = 1000
WRITE_RETRIES = 5


def load_lookup(path: Path, priors: Priors) -> dict[tuple[str, str], tuple[str, int] | None]:
    """Load unique `(kind, surface)` aliases, omitting functional entities."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT surface, kind, eid, node_id FROM alias").fetchall()
    by_entity: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_entity[row["eid"]].append(row)

    functional_entities = {
        eid
        for eid, aliases in by_entity.items()
        if any(
            alias["kind"] == "email"
            and priors.is_functional(alias["surface"].partition("@")[0])
            for alias in aliases
        )
    }
    candidates: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        if row["eid"] not in functional_entities:
            candidates[(row["kind"], row["surface"].lower())].add(
                (row["eid"], int(row["node_id"]))
            )
    return {
        key: next(iter(records)) if len(records) == 1 else None
        for key, records in candidates.items()
    }


def structured_surfaces(doc: dict, priors: Priors) -> Iterable[tuple[str, str]]:
    """Yield normalized identity surfaces from the documented normalized schema."""
    for value in doc.get("emails") or ():
        if value:
            yield "email", value.lower()
    for pair in doc.get("named_emails") or ():
        if value := pair.get("email"):
            yield "email", value.lower()
        if value := pair.get("name"):
            yield "name", norm_name(value, priors)
    for value in doc.get("speakers") or ():
        if value:
            yield "name", norm_name(value, priors)
    for value in doc.get("mentions") or ():
        if value:
            yield "handle", value.lower()
    for attendee in doc.get("attendees") or ():
        if value := attendee.get("name"):
            yield "name", norm_name(value, priors)


def document_links(
    doc: dict,
    lookup: dict[tuple[str, str], tuple[str, int] | None],
    priors: Priors,
    counts: Counter,
) -> list[dict]:
    """Resolve one document's structured aliases and deduplicate entity edges."""
    resolved: dict[tuple[str, int], dict] = {}
    for kind, surface in structured_surfaces(doc, priors):
        record = lookup.get((kind, surface))
        if record is None:
            counts["ambiguous" if (kind, surface) in lookup else "unresolved"] += 1
            continue
        eid, entity_node = record
        link = resolved.setdefault(
            (eid, entity_node),
            {
                "src": entity_node,
                "dst": node_id(f"doc:{doc['doc_id']}"),
                "mention_count": 0,
                "kinds": set(),
            },
        )
        link["mention_count"] += 1
        link["kinds"].add(kind)
    return [
        {**link, "kinds": ",".join(sorted(link["kinds"]))}
        for link in resolved.values()
    ]


def _checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run(
    sources: list[str],
    limit: int | None = None,
    offset: int = 0,
    *,
    normalized: Path = NORMALIZED,
    lookup_path: Path = LOOKUP,
    checkpoint_path: Path = CHECKPOINT,
    engine: GraphEngine | None = None,
    priors: Priors | None = None,
) -> Counter:
    """Load selected shards and return loader counts for tests and reporting."""
    if not lookup_path.exists():
        raise SystemExit("no ontology.sqlite3; run scripts/load_graph.py first")
    priors = priors or (
        Priors.from_dict(json.loads((STATE / "priors.json").read_text()))
        if (STATE / "priors.json").exists()
        else Priors()
    )
    lookup = load_lookup(lookup_path, priors)
    engine = engine or GraphEngine()
    if not engine.wait_until_ready(90):
        raise SystemExit("engine not reachable; is `docker compose up -d` running?")

    # Explicit slices are reproducible smoke tests. Only an unbounded run uses
    # progress state to resume after an interruption.
    resume = _checkpoint(checkpoint_path) if limit is None and offset == 0 else {}
    resume_source, resume_line = resume.get("source"), int(resume.get("line", 0))
    resume_index = sources.index(resume_source) if resume_source in sources else 0
    same_source_order = resume.get("sources") == sources
    counts: Counter = Counter()
    documents: list[dict] = []
    edges: list[dict] = []
    started = time.time()

    def flush() -> None:
        if not documents:
            return
        for attempt in range(WRITE_RETRIES):
            try:
                for start in range(0, len(documents), BATCH):
                    engine.upsert_nodes(
                        "Document", documents[start : start + BATCH], ["doc_id", "source", "title", "date"]
                    )
                for start in range(0, len(edges), BATCH):
                    engine.merge_edges(
                        "MENTIONED_IN",
                        edges[start : start + BATCH],
                        ["mention_count", "kinds"],
                        src_label="Entity",
                        dst_label="Document",
                    )
                break
            except GraphError:
                if attempt == WRITE_RETRIES - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        documents.clear()
        edges.clear()

    for source_index, source in enumerate(sources):
        if resume and same_source_order and source_index < resume_index:
            continue
        shard = normalized / f"{source}.jsonl"
        if not shard.exists():
            print(f"  {source:14s} missing shard, skipping")
            continue
        start = offset if offset else (resume_line if source == resume_source else 0)
        loaded = 0
        last_line = start
        with shard.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number < start:
                    continue
                if limit is not None and loaded >= limit:
                    break
                doc = json.loads(line)
                last_line = line_number + 1
                doc_id = doc.get("doc_id")
                if not doc_id:
                    counts["invalid_documents"] += 1
                    continue
                links = document_links(doc, lookup, priors, counts)
                documents.append(
                    {
                        "id": node_id(f"doc:{doc_id}"),
                        "doc_id": doc_id,
                        "source": doc.get("source") or source,
                        "title": (doc.get("title") or doc_id)[:400],
                        "date": doc.get("date") or "",
                    }
                )
                edges.extend(links)
                counts["documents"] += 1
                counts["linked_documents"] += bool(links)
                counts["edges"] += len(links)
                loaded += 1
                if len(documents) >= BATCH:
                    flush()
                    if limit is None and offset == 0:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint_path.write_text(
                            json.dumps(
                                {"sources": sources, "source": source, "line": last_line}
                            )
                            + "\n",
                            encoding="utf-8",
                        )
        flush()
        if limit is None and offset == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps({"sources": sources, "source": source, "line": last_line})
                + "\n",
                encoding="utf-8",
            )

    elapsed = max(time.time() - started, 1e-6)
    print(f"  documents            {counts['documents']:,}")
    print(f"  linked documents     {counts['linked_documents']:,}")
    print(f"  entity/document edges {counts['edges']:,}")
    print(f"  ambiguous skipped    {counts['ambiguous']:,}")
    print(f"  unresolved skipped   {counts['unresolved']:,}")
    print(f"  rate                 {counts['documents'] / elapsed:,.0f} docs/s")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", help="normalized source shard to load")
    parser.add_argument("--limit", type=int, default=None, help="maximum records per selected source")
    parser.add_argument("--offset", type=int, default=0, help="record offset per selected source")
    args = parser.parse_args()
    sources = args.source or [p.stem for p in sorted(NORMALIZED.glob("*.jsonl"))]
    if not sources:
        raise SystemExit("no normalized shards; run scripts/intake.py first")
    run(sources, args.limit, args.offset)


if __name__ == "__main__":
    main()
