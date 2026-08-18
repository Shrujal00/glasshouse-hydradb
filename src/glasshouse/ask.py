"""The question path: from a question to a cited answer, through the graph.

Five steps, each of which emits a reasoning event so the whole thing can be
watched rather than merely trusted:

    recall     half a million documents -> ~20 candidates      (local, ~50ms)
    identify   the identity surfaces those documents contain
    resolve    surfaces -> canonical people, via the ontology  (~1ms)
    graph      retrieve direct document neighbors from the offline HydraDB graph
    answer     cited answer, or an honest account of what is missing

Documents and entity links are loaded offline. Query-time code only reads that
graph, so HydraDB can add evidence that lexical retrieval never reached.

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
from typing import Any, Iterable, Sequence

from . import answer
from .config import STATE
from .corpus import parse_document_text
from .priors import Priors
from .graph import GraphCandidate, GraphEngine
from .recall import Candidate, LocalRecall

LOOKUP = STATE / "ontology.sqlite3"
PRIORS = STATE / "priors.json"

# How many people to draw. The canvas shows what the question touched, not the
# corpus, and past roughly this many the picture stops being readable.
TOP_PEOPLE = 8
GRAPH_SCOPE_LIMIT = 200
GRAPH_SEED_LIMIT = 8

# Generic role vocabulary supplements the corpus-learned functional-mailbox
# prior when an ontology alias spells out the role rather than its mailbox.
# A multi-word phrase containing one of these is a desk, a process or a status
# line rather than a person; no real name in this corpus contains them.
_ROLE_WORD = frozenset(
    "admin billing compliance customer finance help onboarding operations ops security support team".split()
    + "approvals contracts engineering incident internal legal platform policy".split()
    + "procurement response runner service unavailable vendor".split()
    + ["lead"]
)


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


_METRIC_TOKEN = re.compile(r"^[a-z]?\d+[a-z]?$")


def _metric_shaped(phrase: str) -> bool:
    """`p95`, `95p`, `429`: a measurement or status code, never a person."""
    return bool(_METRIC_TOKEN.match(phrase))


def _role_alias(phrase: str) -> bool:
    return len(phrase.split()) > 1 and bool(set(phrase.split()) & _ROLE_WORD)


def _name_tokens(phrase: str) -> set[str]:
    """The alphanumeric runs of a surface, so punctuation variants agree."""
    return {t for t in re.split(r"[^a-z0-9]+", phrase.lower()) if len(t) > 2}


def _organizational(canonical_name: str, surfaces: Sequence[tuple[str, str]]) -> bool:
    """True when the evidence describes a company, mailbox or concept.

    An organization owns the domain it is named after, so `acme health` is
    reachable at `acmehealth.com` and `horizon analytics` at `horizonfinance`.
    A concept is usually one phrase punctuated two ways -- `routing policy`
    and `routing_policy` -- which looks like corroboration but is a single
    observed surface. A real person's mail domain has nothing to do with their
    name: `p.nair@heliumhealth.com` is a person, `support@acmehealth.com` is
    the company.

    Judging the localpart instead of the domain does not work, because a
    person's own address repeats their own name too -- `maya.chen@redwood.ai`
    is indistinguishable from `vendor-contracts@datasafe.com` by that test.
    Shared mailboxes on an unrelated domain therefore still get through here
    and are left to the learned functional-mailbox priors.
    """
    tokens = _name_tokens(canonical_name)
    if not tokens:
        return False
    emails = [surface for surface, kind in surfaces if kind == "email"]
    for email in emails:
        _, _, domain = email.partition("@")
        # `acme health` registers `acmehealth.com`, not `acme.health.com`, so
        # the comparison is against the host with its separators and public
        # suffix removed rather than against its labels.
        host = "".join(re.findall(r"[a-z0-9]+", domain.lower())[:-1])
        if host and any(token in host for token in tokens):
            return True
    if emails:
        return False
    # No mailbox at all: demand two genuinely different spellings, which a
    # single multi-word phrase repunctuated does not provide. A one-token
    # handle that is also used as a name stays admissible -- that is how a
    # person known only by a first name appears.
    spellings = {frozenset(_name_tokens(surface)) for surface, _ in surfaces}
    return len(spellings) < 2 and len(tokens) > 1


def document_mentions(person: "Person", document: Candidate) -> bool:
    """Match a resolved surface only when the document parser emitted it."""
    found = parse_document_text(document.text, document.source)
    fields = {
        value.lower()
        for values in found.values()
        for value in values
    }
    return bool(person.surfaces & fields)


@dataclass(slots=True)
class Event:
    """One step of visible reasoning."""

    kind: str
    detail: dict

    def line(self) -> str:
        bits = " ".join(
            f"{key}={value}"
            for key, value in self.detail.items()
            if key not in {"items", "path"}
        )
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
    text: str = ""
    cited: list[int] = field(default_factory=list)

    def render(self) -> str:
        out: list[str] = []
        if self.text and not self.abstained:
            out.extend(("ANSWER", f"  {self.text}"))
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
        if self.cited:
            out.append("\nCITED SOURCES")
            for n in self.cited:
                if 1 <= n <= len(self.documents):
                    out.append(f"  [{n}] {self.documents[n - 1].cite()}")
        if self.paths:
            out.append("\nCONNECTIONS FOUND BY HYDRADB")
            for p in self.paths[:6]:
                out.append(f"  {p['summary']}")
        out.append(f"\n({self.elapsed*1000:.0f}ms)")
        return "\n".join(out)


@dataclass(slots=True)
class RetrievalResult:
    """The independent retrieval ablations retained for inspection and scoring."""

    plain_docs: list[Candidate]
    identity_docs: list[Candidate]
    graph_docs: list[Candidate]
    final_docs: list[Candidate]
    named_entities: list[Person]
    graph_candidates: list[GraphCandidate]
    graph_error: str | None = None


class Asker:
    """Holds the open indexes so a question costs milliseconds, not a rebuild."""

    def __init__(self, engine: GraphEngine | None = None) -> None:
        self.recall = LocalRecall()
        self.engine = engine or GraphEngine()
        self._lookup: sqlite3.Connection | None = None
        self._lookup_path = LOOKUP
        self._local = threading.local()
        # The same learned priors the resolver used. Needed here because a
        # department is a department wherever its name appears: `procurement`
        # was filtered out as a mailbox but sailed through as an @mention and
        # was reported to the user as a person.
        self.priors = (
            Priors.from_dict(json.loads(PRIORS.read_text())) if PRIORS.exists() else Priors()
        )

    @property
    def lookup(self) -> sqlite3.Connection:
        """This thread's ontology connection; see `LocalRecall.conn`.

        `_lookup` stays honoured so a caller can inject an open connection.
        """
        if self._lookup is not None:
            return self._lookup
        local = self.__dict__.setdefault("_local", threading.local())
        conn = getattr(local, "lookup", None)
        if conn is None:
            path = getattr(self, "_lookup_path", LOOKUP)
            if not path.exists():
                raise RuntimeError("no ontology lookup; run scripts/load_graph.py first")
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            local.lookup = conn
        return conn

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
        raw_words = [w for w in re.findall(r"[\w.@'-]+", question) if len(w) > 1]
        words = [w.lower() for w in raw_words]
        seen: dict[str, Person] = {}
        for size in (3, 2, 1):
            for i in range(len(words) - size + 1):
                phrase = " ".join(words[i : i + size])
                raw_phrase = " ".join(raw_words[i : i + size])
                identifier_shaped = _identifier_shaped(phrase)
                capitalized_single = (
                    size == 1 and not identifier_shaped and raw_phrase[:1].isupper()
                )
                if not identifier_shaped and not capitalized_single:
                    continue
                if _role_alias(phrase) or _metric_shaped(phrase):
                    continue
                lookup_phrase = phrase[1:] if size == 1 and phrase.startswith("@") else phrase
                rows = self.lookup.execute(
                    "SELECT DISTINCT eid, node_id, canonical_name, confidence, alias_count"
                    " FROM alias WHERE surface = ? LIMIT 2",
                    (lookup_phrase,),
                ).fetchall()
                if len(rows) != 1:
                    continue
                r = rows[0]
                if r["eid"] in seen:
                    continue
                if capitalized_single:
                    given_name_evidence = self.lookup.execute(
                        "SELECT count(DISTINCT surface) FROM alias "
                        "WHERE kind = 'name' AND surface GLOB ?",
                        (f"{lookup_phrase} *",),
                    ).fetchone()[0]
                    if given_name_evidence < 3:
                        continue
                # Every string the parser ever noticed became an entity, so the
                # table holds 166,429 of them and only 38,853 carry both a
                # second surface form and a personal name. The rest are channel
                # tags, status lines and vendor mailboxes that were never
                # resolved to anything. Personhood has to be demonstrated: the
                # resolver must have collapsed separate surfaces onto this
                # entity, and one of them must be a name. Seeding graph
                # retrieval with `finance` otherwise pulls in hundreds of
                # documents that share nothing but a channel.
                evidence = self.lookup.execute(
                    "SELECT COUNT(DISTINCT surface) AS surfaces, "
                    "COUNT(DISTINCT kind) AS kinds, "
                    "SUM(kind = 'name') AS names FROM alias WHERE eid = ?",
                    (r["eid"],),
                ).fetchone()
                # A name is what separates a person from a vendor mailbox or a
                # channel tag, and corroboration is what separates a resolved
                # person from a string seen once: either the resolver collapsed
                # two spellings onto this entity, or it saw the one spelling
                # used both as a handle and as a name.
                if not evidence["names"] or (
                    evidence["surfaces"] < 2 and evidence["kinds"] < 2
                ):
                    continue
                aliases = self.lookup.execute(
                    "SELECT surface, kind FROM alias WHERE eid = ?", (r["eid"],)
                ).fetchall()
                # Role aliases can have an address too. The resolver learned
                # functional mailbox localparts from the corpus, so reject the
                # entire entity rather than expanding its display-name alias.
                if any(
                    self.priors.is_functional(a["surface"].partition("@")[0])
                    for a in aliases
                    if a["kind"] == "email"
                ):
                    continue
                if _organizational(
                    r["canonical_name"], [(a["surface"], a["kind"]) for a in aliases]
                ):
                    continue
                seen[r["eid"]] = Person(
                    eid=r["eid"],
                    name=r["canonical_name"],
                    node=int(r["node_id"]),
                    confidence=float(r["confidence"]),
                    alias_count=int(r["alias_count"]),
                    surfaces={lookup_phrase},
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

    def retrieve(self, question: str, limit: int = 20) -> RetrievalResult:
        """Retrieve independently by lexical, identity, and graph-scoped paths."""
        named = self.read_identities(question)
        expansion: list[str] = []
        asked_as: list[str] = []
        for person in named:
            expansion.extend(self.surfaces_of(person))
            asked_as.extend(person.surfaces)

        plain_docs = self.recall.search(question, limit=limit)
        identity_docs = (
            self.recall.search(question, limit=limit, also=expansion, drop=asked_as)
            if expansion
            else plain_docs
        )
        graph_candidates: list[GraphCandidate] = []
        graph_error = None
        if named:
            try:
                graph_candidates = self.engine.documents_for_entities(
                    [(person.eid, person.node) for person in named[:GRAPH_SEED_LIMIT]],
                    GRAPH_SCOPE_LIMIT,
                )
            except Exception as exc:
                graph_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        graph_docs = self.recall.search_scoped(
            question, [candidate.doc_id for candidate in graph_candidates], limit=limit, drop=asked_as
        )
        final_docs = list(identity_docs)
        known = {doc.doc_id for doc in final_docs}
        final_docs.extend(doc for doc in graph_docs if doc.doc_id not in known)
        return RetrievalResult(
            plain_docs=plain_docs,
            identity_docs=identity_docs,
            graph_docs=graph_docs,
            final_docs=final_docs,
            named_entities=named,
            graph_candidates=graph_candidates,
            graph_error=graph_error,
        )

    @staticmethod
    def _answer_documents(retrieval: RetrievalResult, limit: int) -> list[Candidate]:
        """Balance both ranked strategies in the bounded synthesis context."""
        documents: list[Candidate] = []
        seen: set[str] = set()
        for rank in range(max(len(retrieval.identity_docs), len(retrieval.graph_docs))):
            for strategy in (retrieval.identity_docs, retrieval.graph_docs):
                if rank >= len(strategy) or strategy[rank].doc_id in seen:
                    continue
                documents.append(strategy[rank])
                seen.add(strategy[rank].doc_id)
                if len(documents) >= limit:
                    return documents
        return documents

    def connect(self, question: str, people: list[Person], docs: list[Candidate]) -> list[dict]:
        """Render verified pairwise paths between entities named in the query.

        These paths explain existing co-occurrence only. They never expand the
        retrieval scope and this method performs no writes.
        """
        found: list[dict] = []
        for i, left in enumerate(people[:4]):
            for right in people[i + 1 : 5]:
                results = self.engine.paths(
                    left.node,
                    right.node,
                    ["MENTIONED_IN"],
                    max_len=2,
                    path_count=2,
                )
                for result in results:
                    path = result.get("path") or {}
                    documents = [
                        node["properties"].get("title")
                        for node in path.get("nodes", [])
                        if isinstance(node, dict)
                        and "title" in (node.get("properties") or {})
                    ]
                    if documents:
                        found.append(
                            {
                                "a": left.name,
                                "b": right.name,
                                "via": documents,
                                "summary": (
                                    f"{left.name} - {documents[0][:52]} - {right.name}"
                                ),
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

        retrieval = self.retrieve(question, limit)
        for person in retrieval.named_entities:
            surfaces = self.surfaces_of(person)
            others = [s for s in surfaces if s not in person.surfaces]
            yield {
                "type": "expanded",
                "asked_as": sorted(person.surfaces)[0],
                "name": person.name,
                "also_known_as": sorted(others)[:8],
            }

        docs = self._answer_documents(retrieval, limit)
        yield {
            "type": "recall",
            "documents": len(docs),
            "terms": self.recall.selective_terms(question),
            "identity_only": len(
                {doc.doc_id for doc in retrieval.identity_docs}
                - {doc.doc_id for doc in retrieval.plain_docs}
            ),
            "ms": round((time.time() - t0) * 1000),
        }
        yield {
            "type": "graph_scope",
            "entities": [person.eid for person in retrieval.named_entities],
            "candidates": len(retrieval.graph_candidates),
            "available": retrieval.graph_error is None,
        }
        if retrieval.graph_error:
            yield {
                "type": "degraded",
                "step": "graph_retrieval",
                "detail": retrieval.graph_error,
            }
        identity_ids = {doc.doc_id for doc in retrieval.identity_docs}
        candidates_by_doc = {candidate.doc_id: candidate for candidate in retrieval.graph_candidates}
        for doc in retrieval.graph_docs:
            if doc.doc_id in identity_ids:
                continue
            candidate = candidates_by_doc.get(doc.doc_id)
            yield {
                "type": "graph_document",
                "doc_id": doc.doc_id,
                "seed_eids": list(candidate.seed_eids) if candidate else [],
                "hops": candidate.hops if candidate else 0,
                "reason": candidate.reason if candidate else "",
                "path": candidate.path if candidate else {},
            }
        yield {
            "type": "graph_ablation",
            "plain": len(retrieval.plain_docs),
            "identity": len(retrieval.identity_docs),
            "graph": len(retrieval.graph_docs),
            "graph_only": len(
                {doc.doc_id for doc in retrieval.graph_docs} - identity_ids
            ),
            "available": retrieval.graph_error is None,
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

        # Which people appear in which documents — the edges the canvas draws.
        for p in people:
            for d in docs:
                if document_mentions(p, d):
                    yield {
                        "type": "link",
                        "from": f"ent:{p.eid}",
                        "to": f"doc:{d.doc_id}",
                    }

        # Pairwise paths explain named entities after retrieval; they do not
        # add collaborators or documents to the candidate scope.
        try:
            paths = self.connect(question, retrieval.named_entities, docs)
        except Exception as exc:
            paths = []
            yield {
                "type": "degraded",
                "step": "traversal",
                "detail": type(exc).__name__,
            }
        for path in paths:
            yield {"type": "path", **path}

        # Now compose the answer, with graph evidence included.
        written: queue.Queue = queue.Queue()

        def compose() -> None:
            try:
                for part in answer.write_streaming(question, docs, people, paths=paths):
                    written.put(part)
            except Exception as exc:
                written.put({"error": f"{type(exc).__name__}: {exc}"})
            written.put(None)

        threading.Thread(target=compose, daemon=True).start()

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

        retrieval = self.retrieve(question, limit)
        docs = self._answer_documents(retrieval, limit)
        events.append(
            Event(
                "recall",
                {
                    "documents": len(docs),
                    "terms": len(self.recall.selective_terms(question)),
                },
            )
        )
        events.append(
            Event(
                "graph_scope",
                {
                    "entities": len(retrieval.named_entities),
                    "candidates": len(retrieval.graph_candidates),
                    "available": retrieval.graph_error is None,
                },
            )
        )
        if retrieval.graph_error:
            events.append(
                Event(
                    "graph_degraded",
                    {"detail": retrieval.graph_error},
                )
            )
        identity_ids = {doc.doc_id for doc in retrieval.identity_docs}
        candidates_by_doc = {candidate.doc_id: candidate for candidate in retrieval.graph_candidates}
        for doc in retrieval.graph_docs:
            candidate = candidates_by_doc.get(doc.doc_id)
            if candidate is not None and doc.doc_id not in identity_ids:
                events.append(
                    Event(
                        "graph_document",
                        {
                            "doc_id": candidate.doc_id,
                            "seed_eids": candidate.seed_eids,
                            "hops": candidate.hops,
                            "reason": candidate.reason,
                            "path": candidate.path,
                        },
                    )
                )
        events.append(
            Event(
                "graph_ablation",
                {
                    "plain": len(retrieval.plain_docs),
                    "identity": len(retrieval.identity_docs),
                    "graph": len(retrieval.graph_docs),
                    "graph_only": len(
                        {doc.doc_id for doc in retrieval.graph_docs} - identity_ids
                    ),
                    "available": retrieval.graph_error is None,
                },
            )
        )
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

        try:
            paths = (
                self.connect(question, retrieval.named_entities, docs)
                if len(retrieval.named_entities) > 1
                else []
            )
        except Exception as exc:
            paths = []
            events.append(Event("paths_degraded", {"detail": type(exc).__name__}))
        events.append(Event("paths_walked", {"connections": len(paths)}))

        abstained = None
        if not people:
            # Documents matched, but nobody in them resolves to a known
            # identity — so we can point at the material without pretending to
            # know who it concerns.
            abstained = "matching documents exist, but no known person resolves within them"

        text = ""
        cited: list[int] = []
        if not abstained:
            try:
                written = answer.write(question, docs, people, paths=paths)
                text = written.text
                cited = written.cited
                if written.abstained:
                    abstained = written.text or "the retrieved documents do not contain the answer"
            except Exception as exc:
                events.append(Event("answer_degraded", {"detail": type(exc).__name__}))
                abstained = f"answer synthesis failed ({type(exc).__name__})"

        return Answer(
            question=question,
            people=people,
            documents=docs,
            paths=paths,
            events=events,
            abstained=abstained,
            elapsed=time.time() - t0,
            text=text,
            cited=cited,
        )
