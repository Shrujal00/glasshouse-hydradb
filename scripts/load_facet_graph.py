#!/usr/bin/env python
"""Load the facet half of the graph: containers, speakers and senders.

    python scripts/load_facet_graph.py --source confluence --limit 2000

`load_document_graph.py` keyed the graph on people -- who is mentioned where.
That entrance only opens when a question names somebody, and measurement says
21 of 570 benchmark questions do. The `metadata` questions instead name a
*place* ("in the internal customer success and support knowledge space", "what
Slack channel hosts the discussion") or a *role* ("who was the internal
organizer", "who sent the attachment"). Both are already in the normalized
records and neither was in the graph:

    (:Container {key, source, kind, name, documents})
    (:Document)-[:IN_CONTAINER]->(:Container)
    (:Entity)-[:SPOKE_IN]->(:Document)     -- speakers: fireflies, slack
    (:Entity)-[:SENT]->(:Document)         -- gmail headers.from

Documents and Entities are expected to exist already -- this loader adds
containers and edges and never creates a Document or an Entity, so a run
against a graph that has not had `load_document_graph.py` applied writes
containers and silently matches no endpoints, which is the same failure the
edge MERGE reports as zero edges.

Entity resolution goes through `load_document_graph.load_lookup`, the same
`(kind, surface)` table the mention edges used, so a surface resolves to one
entity here or nowhere; there is no second resolution path. Only normalized
identity fields participate. Document bodies are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from email.utils import parseaddr
from pathlib import Path
from typing import Callable, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glasshouse.config import NORMALIZED, STATE  # noqa: E402
from glasshouse.graph import GraphEngine, GraphError, node_id  # noqa: E402
from glasshouse.priors import Priors  # noqa: E402
from glasshouse.resolve import norm_name  # noqa: E402
from load_document_graph import LOOKUP, load_lookup  # noqa: E402

CHECKPOINT = STATE / "facet_graph_checkpoint.json"
BATCH = 1000
WRITE_RETRIES = 5

# Which normalized list becomes which kind of container. Both hold names of
# places documents live in, and both are what the metadata questions point at.
CONTAINER_FIELDS = (("folder", "folders"), ("channel", "channels"))


def container_key(source: str, kind: str, name: str) -> str:
    """The stable container key, lowercased.

    `FacetStore` builds the same key from the same three fields and the
    retrieval side turns it into a node id with `node_id(f"container:{key}")`,
    so the case fold has to happen here too: 80 of 7,200 sampled channel names
    carry capitals (`#inc-TENANT123-auth`), and a question naming one in lower
    case must still land on the node the loader wrote.
    """
    return f"{source}:{kind}:{name.strip().lower()}"


def containers_of(doc: dict) -> Iterator[tuple[str, str, str, str]]:
    """Yield `(key, source, kind, name)` for every container a document sits in."""
    source = str(doc.get("source") or "")
    seen: set[str] = set()
    for kind, field in CONTAINER_FIELDS:
        for raw in doc.get(field) or ():
            name = str(raw or "").strip()
            if not name:
                continue
            key = container_key(source, kind, name)
            # A record listing the same folder twice must not count twice
            # against the container's document total.
            if key not in seen:
                seen.add(key)
                yield key, source, kind, name


def speaker_surfaces(doc: dict, priors: Priors) -> Iterator[tuple[str, str]]:
    """Identity surfaces of whoever spoke in this document.

    `structured_surfaces` folds every surface in a record into one stream,
    which is right for "mentioned in" and wrong here: the whole point of
    SPOKE_IN is that it distinguishes the people who talked from the people who
    were talked about. So the roles are read field by field, in the same
    normalized forms `structured_surfaces` produces, and resolved through the
    same lookup.
    """
    for value in doc.get("speakers") or ():
        if value:
            yield "name", norm_name(str(value), priors)


def sender_surfaces(doc: dict, priors: Priors) -> Iterator[tuple[str, str]]:
    """Identity surfaces of the sender, from `headers.from`.

    `Tom Nguyen <tom.nguyen@medispec.com>` carries both halves; the address is
    the unambiguous one, so it is yielded first and the display name only backs
    it up when the address resolves to nothing.
    """
    raw = (doc.get("headers") or {}).get("from")
    if not raw:
        return
    name, address = parseaddr(str(raw))
    if address:
        yield "email", address.lower()
    if name:
        yield "name", norm_name(name, priors)


def entity_links(
    doc: dict,
    surfaces: Iterable[tuple[str, str]],
    lookup: dict[tuple[str, str], tuple[str, int] | None],
    counts: Counter,
    label: str,
) -> list[dict]:
    """Resolve role surfaces to one edge row per entity.

    First-name-only speakers ("Jordan", "Amir") are frequently ambiguous
    across 39,847 people; an ambiguous surface is counted and dropped rather
    than attached to a guess, because a wrong SPOKE_IN edge would read in the
    trace as established authorship.
    """
    resolved: dict[int, dict] = {}
    for kind, surface in surfaces:
        if not surface:
            continue
        record = lookup.get((kind, surface))
        if record is None:
            counts[f"{label}_ambiguous" if (kind, surface) in lookup else f"{label}_unresolved"] += 1
            continue
        _eid, entity_node = record
        link = resolved.setdefault(
            entity_node,
            {
                "src": entity_node,
                "dst": node_id(f"doc:{doc['doc_id']}"),
                "mention_count": 0,
                "kinds": set(),
            },
        )
        link["mention_count"] += 1
        link["kinds"].add(kind)
    return [{**link, "kinds": ",".join(sorted(link["kinds"]))} for link in resolved.values()]


def scan_containers(shard: Path, limit: int | None, start: int = 0) -> dict[str, dict]:
    """Count container membership over the shard before writing any node.

    `documents` has to be the container's real size -- retrieval uses it to
    decide whether a scope is a useful 12 documents or a useless 28,999 -- and
    a resumable loader that accumulated the count as it went would write a
    different number on every restart. So the sizes come from one cheap pass
    over the shard, which makes them identical on a resumed run. With `--limit`
    the count describes the slice that was loaded, nothing more.
    """
    containers: dict[str, dict] = {}
    loaded = 0
    with shard.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if limit is not None and line_number < start:
                continue
            if limit is not None and loaded >= limit:
                break
            doc = json.loads(line)
            if not doc.get("doc_id"):
                continue
            loaded += 1
            for key, source, kind, name in containers_of(doc):
                container = containers.setdefault(
                    key,
                    {
                        "id": node_id(f"container:{key}"),
                        "key": key,
                        "source": source,
                        "kind": kind,
                        "name": name,
                        "documents": 0,
                    },
                )
                container["documents"] += 1
    return containers


def _checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write(call: Callable[[], None]) -> None:
    """Retry a batch through the engine's transient write failures.

    Every write here is a MERGE on a deterministic id, so a batch that
    half-succeeded and was retried overwrites in place instead of doubling.
    """
    for attempt in range(WRITE_RETRIES):
        try:
            call()
            return
        except GraphError:
            if attempt == WRITE_RETRIES - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


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
    """Load containers and role edges for selected shards; return loader counts."""
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

    # Explicit slices are reproducible smoke tests, exactly as in the document
    # loader. Only an unbounded run resumes.
    resume = _checkpoint(checkpoint_path) if limit is None and offset == 0 else {}
    resume_source, resume_line = resume.get("source"), int(resume.get("line", 0))
    resume_index = sources.index(resume_source) if resume_source in sources else 0
    same_source_order = resume.get("sources") == sources
    counts: Counter = Counter()
    memberships: list[dict] = []
    spoke: list[dict] = []
    sent: list[dict] = []
    started = time.time()

    def flush() -> None:
        for rows, rel, props, src_label, dst_label in (
            (memberships, "IN_CONTAINER", ["kind"], "Document", "Container"),
            (spoke, "SPOKE_IN", ["mention_count", "kinds"], "Entity", "Document"),
            (sent, "SENT", ["mention_count", "kinds"], "Entity", "Document"),
        ):
            for start in range(0, len(rows), BATCH):
                chunk = rows[start : start + BATCH]
                _write(
                    lambda rel=rel, chunk=chunk, props=props, src_label=src_label, dst_label=dst_label: engine.merge_edges(
                        rel, chunk, props, src_label=src_label, dst_label=dst_label
                    )
                )
        memberships.clear()
        spoke.clear()
        sent.clear()

    def save(source: str, line: int) -> None:
        if limit is not None or offset:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps({"sources": sources, "source": source, "line": line}) + "\n",
            encoding="utf-8",
        )

    for source_index, source in enumerate(sources):
        if resume and same_source_order and source_index < resume_index:
            continue
        shard = normalized / f"{source}.jsonl"
        if not shard.exists():
            print(f"  {source:14s} missing shard, skipping")
            continue

        start = offset if offset else (resume_line if source == resume_source else 0)
        containers = scan_containers(shard, limit, start)
        counts["containers"] += len(containers)
        rows = sorted(containers.values(), key=lambda row: row["key"])
        for batch_start in range(0, len(rows), BATCH):
            chunk = rows[batch_start : batch_start + BATCH]
            _write(
                lambda chunk=chunk: engine.upsert_nodes(
                    "Container", chunk, ["key", "source", "kind", "name", "documents"]
                )
            )

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
                document_node = node_id(f"doc:{doc_id}")
                for key, _source, kind, _name in containers_of(doc):
                    memberships.append(
                        {"src": document_node, "dst": node_id(f"container:{key}"), "kind": kind}
                    )
                    counts["memberships"] += 1
                spoke_links = entity_links(doc, speaker_surfaces(doc, priors), lookup, counts, "spoke_in")
                sent_links = entity_links(doc, sender_surfaces(doc, priors), lookup, counts, "sent")
                spoke.extend(spoke_links)
                sent.extend(sent_links)
                counts["documents"] += 1
                counts["spoke_in"] += len(spoke_links)
                counts["sent"] += len(sent_links)
                loaded += 1
                if len(memberships) + len(spoke) + len(sent) >= BATCH:
                    flush()
                    save(source, last_line)
        flush()
        save(source, last_line)

    elapsed = max(time.time() - started, 1e-6)
    written = counts["containers"] + counts["memberships"] + counts["spoke_in"] + counts["sent"]
    print(f"  documents            {counts['documents']:,}")
    print(f"  containers           {counts['containers']:,}")
    print(f"  in_container edges   {counts['memberships']:,}")
    print(f"  spoke_in edges       {counts['spoke_in']:,}")
    print(f"  sent edges           {counts['sent']:,}")
    print(f"  speaker ambiguous    {counts['spoke_in_ambiguous']:,}")
    print(f"  speaker unresolved   {counts['spoke_in_unresolved']:,}")
    print(f"  sender ambiguous     {counts['sent_ambiguous']:,}")
    print(f"  sender unresolved    {counts['sent_unresolved']:,}")
    print(f"  elapsed              {elapsed:,.1f}s")
    print(f"  rate                 {written / elapsed * 60:,.0f} items/min ({written:,} written)")
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
