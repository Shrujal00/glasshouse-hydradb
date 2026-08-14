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
import queue
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import answer
from .config import STATE
from .corpus import parse_document_text
from .priors import Priors
from .graph import GraphEngine, node_id
from .recall import Candidate, LocalRecall

LOOKUP = STATE / "ontology.sqlite3"
PRIORS = STATE / "priors.json"

# How many people to draw. The canvas shows what the question touched, not the
# corpus, and past roughly this many the picture stops being readable.
TOP_PEOPLE = 8


def _identifier_shaped(phrase: str) -> bool:
    """Whether a phrase could plausibly be somebody's name or account.

    Resolution is imperfect and a few ordinary words ended up as entities, so
    a bare English word matching the alias table is not evidence that the
    question is about a person — asking about "capacity" was expanding into an
    entity literally named capacity and rewriting the whole search.

    An identity looks like one of three things: several words (a full name),
    an address, or a handle with a separator or digit in it. A single lowercase
    dictionary word is none of those.
    """
    if " " in phrase:
        return True
    return bool(re.search(r"[@._\d-]", phrase))


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

    def read_identities(self, question: str) -> list[Person]:
        """Find the people a question names, by consulting the ontology.

        Every one- two- and three-word run in the question is looked up as a
        possible identity. Three words because that is how far a full name
        plus a qualifier reaches; one because handles and addresses stand
        alone.

        A run that matches more than one person is discarded. `sam` is eight
        different people in this corpus, so expanding it would drag in
        everyone called Sam and drown the question — the ambiguity guard that
        protects resolution protects retrieval for the same reason.
        """
        words = [w for w in re.findall(r"[\w.@'-]+", question.lower()) if len(w) > 1]
        seen: dict[str, Person] = {}
        for size in (3, 2, 1):
            for i in range(len(words) - size + 1):
                phrase = " ".join(words[i : i + size])
                if not _identifier_shaped(phrase):
                    continue
                rows = self.lookup.execute(
                    "SELECT DISTINCT eid, node_id, canonical_name, confidence, alias_count"
                    " FROM alias WHERE surface = ? LIMIT 2",
                    (phrase,),
                ).fetchall()
                if len(rows) != 1:
                    continue
                r = rows[0]
                # A person with one surface form has nothing to expand to.
                if int(r["alias_count"]) < 2 or r["eid"] in seen:
                    continue
                # Require an address. Resolution produced a few entities that
                # are really topics — "Capacity Planning" among them — and a
                # topic dragged into the identity half of the query poisons it.
                # A real person has a mailbox; a subject heading does not.
                has_mail = self.lookup.execute(
                    "SELECT 1 FROM alias WHERE eid = ? AND kind = 'email' LIMIT 1", (r["eid"],)
                ).fetchone()
                if not has_mail:
                    continue
                seen[r["eid"]] = Person(
                    eid=r["eid"],
                    name=r["canonical_name"],
                    node=int(r["node_id"]),
                    confidence=float(r["confidence"]),
                    alias_count=int(r["alias_count"]),
                    surfaces={phrase},
                )
        return list(seen.values())

    def surfaces_of(self, person: Person) -> list[str]:
        """Every surface form of a person, straight out of the lookup."""
        return [
            r[0]
            for r in self.lookup.execute(
                "SELECT DISTINCT surface FROM alias WHERE eid = ?", (person.eid,)
            )
        ]

    def aliases_of(self, person: Person) -> list[dict]:
        """Every surface form that resolves to this person, and why.

        Read out of HydraDB, not out of a file. The evidence lives on the
        `RESOLVES_TO` edge, so asking the graph "what is this person made of?"
        returns the answer and its justification in one traversal — which is
        the whole reason the ontology is a graph rather than a table.
        """
        try:
            rows = self.engine.query(
                "MATCH (a:Alias)-[r:RESOLVES_TO]->(e:Entity {id: $id}) "
                "RETURN a.surface AS surface, a.kind AS kind, "
                "r.score AS score, r.signals AS signals "
                "ORDER BY a.occurrences DESC LIMIT 8",
                {"id": person.node},
                strong=True,
            )
        except Exception:
            return []
        return [
            {
                "surface": r.values.get("surface", ""),
                "kind": r.values.get("kind", ""),
                "score": round(float(r.values.get("score") or 0), 2),
                "signals": r.values.get("signals", ""),
            }
            for r in rows
        ]

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

    def stream(self, question: str, limit: int = 20):
        """The same work as `ask`, yielded step by step as it happens.

        Emitted as graph fragments rather than log lines - each event carries
        the nodes and edges it just established - so a viewer can draw the
        reasoning as it unfolds instead of being shown a finished picture and
        asked to take it on trust.
        """
        t0 = time.time()
        yield {"type": "start", "question": question}

        # Before searching, ask the ontology who this question is about, and
        # search for every name they answer to. This is the step keyword search
        # cannot perform: the question says "Sam", the document says
        # "S. Ratnaparkhi", and only the graph knows they are one person.
        named = self.read_identities(question)
        expansion: list[str] = []
        asked_as: list[str] = []
        for person in named:
            surfaces = self.surfaces_of(person)
            others = [s for s in surfaces if s not in person.surfaces]
            expansion.extend(surfaces)
            asked_as.extend(person.surfaces)
            yield {
                "type": "expanded",
                "asked_as": sorted(person.surfaces)[0],
                "name": person.name,
                "also_known_as": sorted(others)[:8],
            }

        plain = self.recall.search(question, limit=limit)
        docs = (
            self.recall.search(question, limit=limit, also=expansion, drop=asked_as)
            if expansion else plain
        )
        found_only_by_identity = len({d.doc_id for d in docs} - {d.doc_id for d in plain})
        yield {
            "type": "recall",
            "documents": len(docs),
            "terms": self.recall.selective_terms(question),
            "identity_only": found_only_by_identity,
            "ms": round((time.time() - t0) * 1000),
        }
        if not docs:
            yield {
                "type": "abstain",
                "reason": "Nothing in the corpus matches the terms of this question.",
                "gate": "linking",
                "ms": round((time.time() - t0) * 1000),
            }
            return

        for d in docs:
            yield {
                "type": "document",
                "id": f"doc:{d.doc_id}",
                "title": d.title or d.doc_id,
                "source": d.source,
                "date": d.date,
            }

        surfaces = self.identify(docs)
        yield {"type": "surfaces", "count": len(surfaces)}

        everyone = self.resolve(surfaces)
        # Draw only the people the question is actually about. A question can
        # brush against fifty names; rendering all of them produces a hairball
        # that shows the reader nothing, which is the failure this canvas
        # exists to avoid.
        people = everyone[:TOP_PEOPLE]

        for p in people:
            yield {
                "type": "entity",
                "id": f"ent:{p.eid}",
                "name": p.name,
                "confidence": round(p.confidence, 2),
                "alias_count": p.alias_count,
                "surfaces": sorted(p.surfaces),
                "mentions": p.mentions,
            }
            # The aliases this person was assembled from, each carrying the
            # evidence that attached it. This is the thing worth watching:
            # scattered surface forms collapsing into one human being.
            for alias in self.aliases_of(p):
                yield {
                    "type": "alias",
                    "id": f"alias:{p.eid}:{alias['surface']}",
                    "entity": f"ent:{p.eid}",
                    "surface": alias["surface"],
                    "kind": alias["kind"],
                    "signals": alias["signals"],
                    "score": alias["score"],
                }

        yield {
            "type": "resolved",
            "people": len(everyone),
            "shown": len(people),
            "collapsed": sum(1 for p in everyone if p.alias_count > 1),
            "ms": round((time.time() - t0) * 1000),
        }

        # Start writing the answer now, on a background thread, so the model
        # composes while the graph is still being drawn. It must not wait on the
        # identity work or be skipped by it: most questions are not about people
        # at all — "what are the upload size limits" needs the documents alone.
        written: queue.Queue = queue.Queue()

        def compose() -> None:
            try:
                for part in answer.write_streaming(question, docs, people):
                    written.put(part)
            except Exception as exc:
                written.put({"error": f"{type(exc).__name__}: {exc}"})
            written.put(None)

        threading.Thread(target=compose, daemon=True).start()

        # Which people appear in which documents — the edges the canvas draws.
        for p in people:
            for d in docs:
                if any(s in d.text.lower() for s in p.surfaces):
                    yield {
                        "type": "link",
                        "from": f"ent:{p.eid}",
                        "to": f"doc:{d.doc_id}",
                    }

        # Traversal is the last step and the most fragile, because it writes to
        # the engine before it reads. If it fails, the people and the evidence
        # are already established and worth showing — losing the whole answer
        # over a failed graph write would be the wrong trade.
        try:
            paths = self.connect(question, people, docs)
        except Exception as exc:
            paths = []
            yield {
                "type": "degraded",
                "step": "traversal",
                "detail": type(exc).__name__,
            }
        for path in paths:
            yield {"type": "path", **path}

        yield {"type": "writing"}
        cited: list[int] = []
        while True:
            part = written.get()
            if part is None:
                break
            if "chunk" in part:
                yield {"type": "answer_chunk", "text": part["chunk"]}
            elif "done" in part:
                w = part["done"]
                cited = w.cited
                yield {
                    "type": "answer",
                    "text": w.text,
                    "abstained": w.abstained,
                    "cited": w.cited,
                    # Which retrieved document each citation points at, so the
                    # reader can check the claim rather than trust it.
                    "sources": [
                        {"n": n, "cite": docs[n - 1].cite(), "source": docs[n - 1].source}
                        for n in w.cited
                        if 1 <= n <= len(docs)
                    ],
                }
            elif "error" in part:
                yield {"type": "degraded", "step": "answer", "detail": part["error"][:140]}

        yield {
            "type": "done",
            "people": len(people),
            "documents": len(docs),
            "paths": len(paths),
            "ms": round((time.time() - t0) * 1000),
        }

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
