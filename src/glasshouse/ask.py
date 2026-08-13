"""The question path: from a question to a cited answer, through the graph.

Five steps, each of which emits a reasoning event so the whole thing can be
watched rather than merely trusted:

    recall     half a million documents -> ~20 candidates      (local, ~50ms)
    identify   the identity surfaces those documents contain
    resolve    surfaces -> canonical people, via the ontology  (~1ms)
    connect    write what this question touched into HydraDB,
               then ask HydraDB how the entities connect
    answer     cited answer, or an honest account of what is missing

The ontology grows as it is used. Documents and their entity links are written
into the graph at question time, for the handful of documents a question
actually reaches, rather than by a batch pass over the whole corpus. That means
the system answers its first question seconds after the index exists, and the
graph accumulates exactly the parts of the corpus anybody asked about.

What resolution *cannot* be done locally is the ambiguity check. Whether
`@priya` names one person depends on how many Priyas exist across all 500k
documents, not on the twenty in front of us — so that knowledge comes from the
prebuilt ontology lookup, which is small, and is why resolution stays a
microsecond operation instead of a corpus scan.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import STATE
from .corpus import parse_document_text
from .priors import Priors
from .graph import GraphEngine, node_id
from .recall import Candidate, LocalRecall

LOOKUP = STATE / "ontology.sqlite3"
PRIORS = STATE / "priors.json"


@dataclass(slots=True)
class Event:
    """One step of visible reasoning."""

    kind: str
    detail: dict

    def line(self) -> str:
        bits = " ".join(f"{k}={v}" for k, v in self.detail.items() if k != "items")
        return f"{self.kind:18s} {bits}"


@dataclass(slots=True)
class Person:
    """A canonical person the question reached."""

    eid: str
    name: str
    node: int
    confidence: float
    alias_count: int
    surfaces: set[str] = field(default_factory=set)
    mentions: int = 0


@dataclass(slots=True)
class Answer:
    question: str
    people: list[Person]
    documents: list[Candidate]
    paths: list[dict]
    events: list[Event]
    abstained: str | None = None
    elapsed: float = 0.0

    def render(self) -> str:
        out: list[str] = []
        if self.abstained:
            out.append(f"NOT ANSWERABLE FROM THE CORPUS — {self.abstained}")
            if self.documents:
                out.append("\nClosest related material:")
        else:
            out.append("PEOPLE INVOLVED")
            for p in self.people[:8]:
                alias = f"{p.alias_count} aliases" if p.alias_count > 1 else "1 surface"
                out.append(
                    f"  {p.name:32s} {alias:12s} conf {p.confidence:.2f}  "
                    f"seen as {', '.join(sorted(p.surfaces)[:4])}"
                )
            out.append("\nEVIDENCE")
        for d in self.documents[:6]:
            out.append(f"  [{d.source}] {d.cite()}")
        if self.paths:
            out.append("\nCONNECTIONS FOUND BY HYDRADB")
            for p in self.paths[:6]:
                out.append(f"  {p['summary']}")
        out.append(f"\n({self.elapsed*1000:.0f}ms)")
        return "\n".join(out)


class Asker:
    """Holds the open indexes so a question costs milliseconds, not a rebuild."""

    def __init__(self, engine: GraphEngine | None = None) -> None:
        self.recall = LocalRecall()
        self.engine = engine or GraphEngine()
        self._lookup: sqlite3.Connection | None = None
        # The same learned priors the resolver used. Needed here because a
        # department is a department wherever its name appears: `procurement`
        # was filtered out as a mailbox but sailed through as an @mention and
        # was reported to the user as a person.
        self.priors = (
            Priors.from_dict(json.loads(PRIORS.read_text())) if PRIORS.exists() else Priors()
        )

    @property
    def lookup(self) -> sqlite3.Connection:
        if self._lookup is None:
            if not LOOKUP.exists():
                raise RuntimeError("no ontology lookup; run scripts/load_graph.py first")
            self._lookup = sqlite3.connect(LOOKUP)
            self._lookup.row_factory = sqlite3.Row
        return self._lookup

    # --- steps ---------------------------------------------------------------

    def identify(self, docs: Iterable[Candidate]) -> Counter:
        """Every identity surface appearing in the retrieved documents."""
        surfaces: Counter = Counter()
        for d in docs:
            found = parse_document_text(d.text, d.source)
            for value in found["emails"]:
                surfaces[("email", value.lower())] += 1
            for value in found["names"]:
                surfaces[("name", value.lower())] += 1
            for value in found["handles"]:
                surfaces[("handle", value.lower())] += 1
        return surfaces

    def resolve(self, surfaces: Counter) -> list[Person]:
        """Map surfaces onto canonical people using the prebuilt ontology."""
        people: dict[str, Person] = {}
        for (kind, value), n in surfaces.items():
            if self.priors.is_functional(value):
                continue
            rows = self.lookup.execute(
                "SELECT eid, node_id, canonical_name, confidence, alias_count"
                " FROM alias WHERE surface = ? AND kind = ?",
                (value, kind),
            ).fetchall()
            # A surface that resolves to more than one entity is ambiguous and
            # names nobody; silently taking the first would be exactly the
            # confident guess this whole system exists to avoid.
            if len(rows) != 1:
                continue
            r = rows[0]
            p = people.get(r["eid"])
            if p is None:
                p = people[r["eid"]] = Person(
                    eid=r["eid"],
                    name=r["canonical_name"],
                    node=int(r["node_id"]),
                    confidence=float(r["confidence"]),
                    alias_count=int(r["alias_count"]),
                )
            p.surfaces.add(value)
            p.mentions += n
        return sorted(people.values(), key=lambda p: -p.mentions)

    def connect(self, question: str, people: list[Person], docs: list[Candidate]) -> list[dict]:
        """Write what this question touched into HydraDB, then ask it what connects.

        The write is the point as much as the read: after this call the graph
        holds these documents, these people and the links between them, so the
        next question about any of them traverses an ontology that has already
        grown.
        """
        doc_rows = [
            {
                "id": node_id(f"doc:{d.doc_id}"),
                "doc_id": d.doc_id,
                "source": d.source,
                "title": (d.title or d.doc_id)[:400],
                "date": d.date or "",
            }
            for d in docs
        ]
        self.engine.upsert_nodes("Document", doc_rows, ["doc_id", "source", "title", "date"])

        by_doc = {d.doc_id: d for d in docs}
        edges = []
        for person in people:
            for d in docs:
                if any(s in d.text.lower() for s in person.surfaces):
                    edges.append(
                        {
                            "src": person.node,
                            "dst": node_id(f"doc:{d.doc_id}"),
                            "count": person.mentions,
                        }
                    )
        # MERGE, not CREATE: the same question asked twice must not double
        # the graph it grew the first time.
        self.engine.merge_edges(
            "MENTIONED_IN", edges, ["count"], src_label="Entity", dst_label="Document"
        )

        # Now the graph question: which of these people are connected, and
        # through what? Bounded multi-hop, returned whole with properties.
        found: list[dict] = []
        for i, a in enumerate(people[:4]):
            for b in people[i + 1 : 5]:
                for path in self.engine.paths(a.node, b.node, ["MENTIONED_IN"], max_len=2, path_count=2):
                    hops = path.get("path") or {}
                    docs_on_path = [
                        n["properties"].get("title")
                        for n in hops.get("nodes", [])
                        if isinstance(n, dict) and "title" in (n.get("properties") or {})
                    ]
                    if docs_on_path:
                        found.append(
                            {
                                "a": a.name,
                                "b": b.name,
                                "via": docs_on_path,
                                "summary": f"{a.name} ─ {docs_on_path[0][:52]} ─ {b.name}",
                            }
                        )
        return found

    # --- the whole thing -----------------------------------------------------

    def ask(self, question: str, limit: int = 20) -> Answer:
        t0 = time.time()
        events: list[Event] = []

        docs = self.recall.search(question, limit=limit)
        events.append(Event("recall", {"documents": len(docs), "terms": len(self.recall.selective_terms(question))}))
        if not docs:
            return Answer(
                question, [], [], [], events,
                abstained="nothing in the corpus matches the terms of the question",
                elapsed=time.time() - t0,
            )

        surfaces = self.identify(docs)
        events.append(Event("surfaces_found", {"distinct": len(surfaces)}))

        people = self.resolve(surfaces)
        events.append(
            Event(
                "entities_resolved",
                {
                    "people": len(people),
                    "collapsed": sum(1 for p in people if p.alias_count > 1),
                },
            )
        )

        paths = self.connect(question, people, docs) if people else []
        events.append(Event("paths_walked", {"connections": len(paths)}))

        abstained = None
        if not people:
            # Documents matched, but nobody in them resolves to a known
            # identity — so we can point at the material without pretending to
            # know who it concerns.
            abstained = "matching documents exist, but no known person resolves within them"

        return Answer(
            question=question,
            people=people,
            documents=docs,
            paths=paths,
            events=events,
            abstained=abstained,
            elapsed=time.time() - t0,
        )
