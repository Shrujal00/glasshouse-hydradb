"""The question path: from a question to a cited answer, through the graph.

Five steps, each of which emits a reasoning event so the whole thing can be
watched rather than merely trusted:

    recall     half a million documents -> ~20 candidates      (local, ~50ms)
    identify   the identity surfaces those documents contain
    resolve    surfaces -> canonical people, by traversal in HydraDB
    graph      three entrances into the graph, run independently
    answer     cited answer, or an honest account of what is missing

Resolution is a graph traversal. A word from the question anchors a `Surface`
node by its deterministic id, one `DENOTES` hop reaches every person that form
could mean, and *how many it reaches* is the ambiguity guard -- a form that
denotes two people has named neither.

The ambiguity check is the part that cannot be done from the twenty documents
in front of us: whether `@priya` names one person depends on all 500k. That
knowledge is counted once when the graph is loaded and carried on the Surface
node, so the check stays a single anchored hop rather than a corpus scan. The
engine rejects unanchored scans outright, which is what forces the design and,
having forced it, is what makes it fast.
"""

from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import answer, claims as claim_extraction, trust
from .config import STATE
from .corpus import parse_document_text
from .facets import FACETS, Container, DocumentFacets, FacetStore, rerank
from .graph import node_id
from .priors import Priors
from .graph import EntityCandidate, GraphCandidate, GraphEngine, SurfaceMatch
from .recall import Candidate, LocalRecall

PRIORS = STATE / "priors.json"

# How much of the retrieval budget a named container may claim. A question that
# names a folder is usually asking *about* that folder, but the container is a
# scope, not an answer: keyword ranking still has to pick within it, and the
# lexical pool has to survive a container that turned out to be the wrong one.
CONTAINER_SCOPE_LIMIT = 400

# How deep to read before reranking. Measured over 30 `metadata` questions, the
# expected document reaches the top 20 twelve times on BM25 alone, fifteen when
# a 200-document page is rescored against the recorded metadata, and seventeen
# from a 500-document page. Reading deeper than the answer needs costs one
# indexed facet lookup over the page and nothing else.
DEEP_PAGE = 500

# How many people to draw. The canvas shows what the question touched, not the
# corpus, and past roughly this many the picture stops being readable.
TOP_PEOPLE = 8
GRAPH_SCOPE_LIMIT = 200
GRAPH_SEED_LIMIT = 8

# How many distinct surfaces from the retrieved documents to resolve. Identity
# resolution is a graph traversal at roughly 0.09s each and the engine has no
# batched form, so this is the difference between a bounded second and an
# unbounded ten. `TOP_PEOPLE` is 8; resolving 300 surfaces to draw 8 was waste.
RESOLVE_BUDGET = 48
RESOLVE_WORKERS = 8

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
    connected_entities: list[EntityCandidate] = field(default_factory=list)
    connected_error: str | None = None
    # The third entrance: the containers the question named, the documents they
    # hold, and which of the two scopes actually produced them.
    containers: list[Container] = field(default_factory=list)
    container_docs: list[Candidate] = field(default_factory=list)
    container_entrance: str | None = None
    container_error: str | None = None
    source_hint: str | None = None


class Asker:
    """Holds the open indexes so a question costs milliseconds, not a rebuild."""

    def __init__(self, engine: GraphEngine | None = None) -> None:
        self.recall = LocalRecall()
        self.engine = engine or GraphEngine()
        self._local = threading.local()
        # The same learned priors the resolver used. Needed here because a
        # department is a department wherever its name appears: `procurement`
        # was filtered out as a mailbox but sailed through as an @mention and
        # was reported to the user as a person.
        self.priors = (
            Priors.from_dict(json.loads(PRIORS.read_text())) if PRIORS.exists() else Priors()
        )
        self._facets_path = FACETS
        # Resolution is a graph traversal now, and the engine answers about
        # twenty of them a second. A question probes the same word from several
        # phrase lengths and `resolve` asks again for the surfaces retrieval
        # found, so the same form is looked up repeatedly within one question
        # and across a session. Memoised per Asker; a stale entry would need
        # the ontology to be reloaded underneath a running server.
        self._denoted: dict[str, list[SurfaceMatch]] = {}
        self._forms: dict[int, list[tuple[str, str]]] = {}

    @property
    def facets(self) -> FacetStore | None:
        """This thread's facet store, or None while it has not been built.

        Thread-local for the same reason `lookup` is: one process-wide SQLite
        connection on FastAPI's threadpool raises "SQLite objects created in a
        thread can only be used in that same thread" intermittently.

        Absent rather than fatal. The store is an addition to retrieval and
        synthesis, not a dependency of either, so a deployment that has not run
        `scripts/build_facets.py` answers exactly as it did before rather than
        refusing to answer at all.
        """
        path = getattr(self, "_facets_path", FACETS)
        if not path.exists():
            return None
        local = self.__dict__.setdefault("_local", threading.local())
        store = getattr(local, "facets", None)
        if store is None:
            try:
                store = FacetStore(path)
                store.conn  # opened here so a corrupt file fails here, not mid-question
            except Exception:
                return None
            local.facets = store
        return store

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

    def denoted_by(self, text: str) -> list[SurfaceMatch]:
        """Who this written form denotes, from HydraDB, memoised."""
        word = (text or "").strip().lower()
        # `setdefault` rather than the attribute, so an Asker assembled field
        # by field — as the tests do — still memoises instead of raising.
        memo = self.__dict__.setdefault("_denoted", {})
        if word not in memo:
            memo[word] = self.engine.denoted_by(word)
        return memo[word]

    def written_forms(self, entity_node: int) -> list[tuple[str, str]]:
        """Every `(form, kind)` denoting this person, from HydraDB, memoised."""
        memo = self.__dict__.setdefault("_forms", {})
        if entity_node not in memo:
            memo[entity_node] = self.engine.surfaces_of(entity_node)
        return memo[entity_node]

    def _is_a_person(self, match: SurfaceMatch) -> bool:
        """Whether the entity behind a match is a person at all.

        Every string the parser ever noticed became an entity, so most of them
        are channel tags, status lines and vendor mailboxes. Personhood has to
        be demonstrated: the resolver must have collapsed separate spellings
        onto this entity, or seen one spelling used both as a handle and as a
        name — and one of those spellings must be a name. Seeding retrieval
        with `finance` otherwise drags in every document sharing a channel.
        """
        forms = self.written_forms(match.node)
        if not forms:
            return False
        kinds = {kind for _, kind in forms}
        if "name" not in kinds:
            return False
        if len(forms) < 2 and len(kinds) < 2:
            return False
        # A role alias can have an address too, and the resolver learned which
        # localparts are functional from the corpus. Reject the whole entity
        # rather than expanding its display-name spelling.
        if any(
            self.priors.is_functional(text.partition("@")[0])
            for text, kind in forms
            if kind == "email"
        ):
            return False
        return not _organizational(match.name, forms)

    def resolve(self, surfaces: Counter) -> list[Person]:
        """Map surfaces onto canonical people, by traversal through HydraDB.

        Bounded, because every surface now costs a round trip. Twenty documents
        carry hundreds of identity surfaces and the canvas draws eight people,
        so resolving all of them spent seconds to display a handful. The
        commonest are taken first: a surface appearing once in one document is
        the least likely to be who the question is about.
        """
        wanted = surfaces.most_common(RESOLVE_BUDGET)
        # Warm the memo concurrently first. The engine only partly overlaps
        # these, so the win is modest -- but the loop below would otherwise
        # spend the whole budget strictly one round trip at a time.
        cold = [value for (_, value), _ in wanted
                if value.strip().lower() not in self.__dict__.setdefault("_denoted", {})]
        if len(cold) > 1:
            with ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as pool:
                list(pool.map(self.denoted_by, cold))

        people: dict[str, Person] = {}
        for (kind, value), n in wanted:
            if self.priors.is_functional(value):
                continue
            matches = [m for m in self.denoted_by(value) if not kind or kind in m.kinds]
            # A form that denotes more than one person names nobody; silently
            # taking the first would be exactly the confident guess this whole
            # system exists to avoid. The graph reports the count on the
            # surface itself, so ambiguity is a property of the word rather
            # than something inferred from how many rows came back.
            if len(matches) != 1 or matches[0].entities != 1:
                continue
            m = matches[0]
            p = people.get(m.eid)
            if p is None:
                p = people[m.eid] = Person(
                    eid=m.eid,
                    name=m.name,
                    node=m.node,
                    confidence=m.confidence,
                    alias_count=m.alias_count,
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
                matches = self.denoted_by(lookup_phrase)
                # One hop into HydraDB answers both halves at once: who this
                # form denotes, and how many people it could mean. A form that
                # reaches nobody is not a name; a form that reaches several
                # names nobody, because expanding `sam` would drag in every
                # Sam in the corpus and drown the question.
                if len(matches) != 1 or matches[0].entities != 1:
                    continue
                m = matches[0]
                if m.eid in seen:
                    continue
                if capitalized_single:
                    # A bare capitalised word is only a person if the ontology
                    # knows full names beginning with it. The count rides on
                    # the Surface node, computed over the whole ontology when
                    # the graph was loaded, because the engine answers
                    # anchored traversals and rejects the prefix scan that
                    # would work it out here.
                    if m.given_name_forms < 3:
                        continue
                if not self._is_a_person(m):
                    continue
                seen[m.eid] = Person(
                    eid=m.eid,
                    name=m.name,
                    node=m.node,
                    confidence=m.confidence,
                    alias_count=m.alias_count,
                    surfaces={lookup_phrase},
                )
        return list(seen.values())

    def surfaces_of(self, person: Person) -> list[str]:
        """Every written form of a person — one inbound hop through HydraDB."""
        return [text for text, _ in self.written_forms(person.node)]

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

        plain_docs = self.lexical(question, limit)
        identity_docs = (
            self.lexical(question, limit, also=expansion, drop=asked_as)
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
        # The third entrance. A question like "in the internal customer success
        # and support knowledge space" names no person and no keyword the body
        # is likely to repeat -- it names a *place*. Folders and channels are
        # that place. `containers_named` is deliberately hard to satisfy: one
        # ordinary word is never enough, because `engineering` is 21,841
        # documents and scoping to it is the same as not scoping at all.
        store = self.facets
        containers: list[Container] = []
        container_docs: list[Candidate] = []
        container_entrance = None
        container_error = None
        source_hint = None
        if store is not None:
            try:
                source_hint = store.source_hint(question)
                containers = store.containers_named(question)
                if source_hint:
                    # Container names repeat across sources -- a Slack channel,
                    # a Drive folder and a Confluence space all called
                    # `security-review` -- so a question that said which source
                    # it meant has already ruled the other two out. Measured on
                    # 30 metadata questions: without this the entrance fired on
                    # 18 and spent most of its scope on channels belonging to a
                    # source the question had excluded.
                    kept = [c for c in containers if c.source == source_hint]
                    # An empty result means the match was in the wrong source
                    # entirely, which is a miss, not a reason to fall back to
                    # it: the fallback is ordinary keyword retrieval, and it is
                    # still running.
                    containers = kept
            except Exception as exc:
                container_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        if containers:
            scope: list[str] = []
            try:
                # HydraDB first, so the container hop is a graph hop when the
                # graph holds it. `documents_in_containers` fails soft while
                # `scripts/load_facet_graph.py` has not run, and the local
                # store below serves the identical scope in the meantime --
                # the trace records which of the two answered.
                reached = self.engine.documents_in_containers(
                    [(c.key, node_id(f"container:{c.key}")) for c in containers],
                    CONTAINER_SCOPE_LIMIT,
                )
                scope = [candidate.doc_id for candidate in reached]
                container_entrance = "hydradb" if scope else None
            except Exception as exc:
                container_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            if not scope and store is not None:
                try:
                    scope = store.documents_in(
                        [c.key for c in containers], CONTAINER_SCOPE_LIMIT
                    )
                    container_entrance = "facets" if scope else None
                except Exception as exc:
                    container_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            if scope:
                container_docs = self.recall.search_scoped(
                    question, scope, limit=limit, drop=asked_as
                )

        final_docs = list(identity_docs)
        known = {doc.doc_id for doc in final_docs}
        final_docs.extend(doc for doc in graph_docs if doc.doc_id not in known)
        known.update(doc.doc_id for doc in graph_docs)
        final_docs.extend(doc for doc in container_docs if doc.doc_id not in known)

        # The other way into the graph. Seeding from people named in the
        # question opens for 21 of 570 benchmark questions and never for "who
        # owns X", where the person is the answer. Reading MENTIONED_IN inwards
        # from the documents retrieval already found gives the people attached
        # to that evidence, ranked by how much of it they are attached to.
        connected: list[EntityCandidate] = []
        connected_error = None
        if final_docs:
            try:
                connected = self.engine.entities_for_documents(
                    [doc.doc_id for doc in final_docs], GRAPH_SCOPE_LIMIT
                )
            except Exception as exc:
                # Its own failure. The forward scope reports separately, so a
                # reverse traversal that breaks does not claim the whole graph
                # is unreachable.
                connected_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        return RetrievalResult(
            plain_docs=plain_docs,
            identity_docs=identity_docs,
            graph_docs=graph_docs,
            final_docs=final_docs,
            named_entities=named,
            graph_candidates=graph_candidates,
            graph_error=graph_error,
            connected_entities=connected,
            connected_error=connected_error,
            containers=containers,
            container_docs=container_docs,
            container_entrance=container_entrance,
            container_error=container_error,
            source_hint=source_hint,
        )

    @staticmethod
    def _answer_documents(retrieval: RetrievalResult, limit: int) -> list[Candidate]:
        """Balance the ranked strategies in the bounded synthesis context.

        Round-robin rather than concatenation, and the container scope goes
        first when it fired. Half the failing `metadata` questions never
        retrieved their document at all; the ones the container scope rescues
        are precisely the ones keyword ranking put outside the top twenty, so
        appending them behind two full keyword lists would retrieve them and
        then crowd them straight back out of the six-document context.
        """
        strategies = [retrieval.identity_docs, retrieval.graph_docs]
        if retrieval.container_docs:
            strategies.insert(0, retrieval.container_docs)
        documents: list[Candidate] = []
        seen: set[str] = set()
        for rank in range(max((len(s) for s in strategies), default=0)):
            for strategy in strategies:
                if rank >= len(strategy) or strategy[rank].doc_id in seen:
                    continue
                documents.append(strategy[rank])
                seen.add(strategy[rank].doc_id)
                if len(documents) >= limit:
                    return documents
        return documents

    # --- claims --------------------------------------------------------------

    def adjudicate(self, question: str, docs: list[Candidate]) -> trust.Arbitration:
        """Extract the claims the evidence states, then arbitrate between them.

        Retrieval is not the bottleneck for `conflicting_info`: plain FTS puts
        the expected document in the context on 10 of 10 of those questions.
        The model still scores about half, because it can see two competing
        values and has no basis for preferring one. This gives it one --
        recency, source authority, explicitness, corroboration -- and, when the
        margin is too thin to justify a choice, tells it to report both rather
        than invent a winner.

        Never raises: both halves degrade to no claims, and no claims is the
        behaviour that exists today.
        """
        try:
            found = claim_extraction.extract(docs, question)
            return trust.arbitrate(found)
        except Exception:
            return trust.Arbitration(claims=(), conflicts=())

    def lexical(
        self, question: str, limit: int, also: Sequence[str] = (), drop: Sequence[str] = ()
    ) -> list[Candidate]:
        """Keyword retrieval, reranked against what the documents record.

        Reads `DEEP_PAGE` documents and returns `limit`. With no facet store
        this is exactly `recall.search` at `limit`, which is what it was before
        the store existed.
        """
        store = self.facets
        depth = max(limit, DEEP_PAGE) if store is not None else limit
        found = self.recall.search(question, limit=depth, also=also, drop=drop)
        if store is None or not found:
            return found[:limit]
        try:
            recorded = store.facets_for([doc.doc_id for doc in found])
        except Exception:
            return found[:limit]
        return rerank(question, found, recorded, limit=limit)

    def _facets_for(self, docs: list[Candidate]) -> dict[str, DocumentFacets]:
        """The recorded metadata for the documents synthesis will read."""
        store = self.facets
        if store is None or not docs:
            return {}
        try:
            return store.facets_for([doc.doc_id for doc in docs])
        except Exception:
            return {}

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

        # Claim extraction is a model call, so it starts now and is collected
        # just before composing. The graph steps below take roughly as long,
        # and running the two in sequence would have shown the reader a stall.
        adjudged: dict[str, trust.Arbitration] = {}

        def adjudicating() -> None:
            adjudged["result"] = self.adjudicate(question, docs[: answer.MAX_DOCS])

        arbiter = threading.Thread(target=adjudicating, daemon=True)
        arbiter.start()

        facets = self._facets_for(docs[: answer.MAX_DOCS])
        yield {
            "type": "recall",
            "documents": len(docs),
            "matched": self.recall.match_count(question),
            "terms": self.recall.selective_terms(question),
            "identity_only": len(
                {doc.doc_id for doc in retrieval.identity_docs}
                - {doc.doc_id for doc in retrieval.plain_docs}
            ),
            "ms": round((time.time() - t0) * 1000),
        }
        if retrieval.containers:
            yield {
                "type": "containers",
                "named": [
                    {"key": c.key, "kind": c.kind, "source": c.source,
                     "documents": c.documents}
                    for c in retrieval.containers
                ],
                # Which scope answered, so the trace does not imply a graph hop
                # that the local table actually served.
                "entrance": retrieval.container_entrance,
                "documents": len(retrieval.container_docs),
                "only": len(
                    {doc.doc_id for doc in retrieval.container_docs}
                    - {doc.doc_id for doc in retrieval.plain_docs}
                ),
                "available": retrieval.container_error is None,
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
        corroborated = [e for e in retrieval.connected_entities if e.documents > 1]
        for entity in (corroborated or retrieval.connected_entities[:5])[:8]:
            yield {
                "type": "connected_entity",
                "eid": entity.eid,
                "name": entity.name,
                "documents": entity.documents,
                "reason": entity.reason,
            }
        if retrieval.connected_error:
            yield {
                "type": "degraded",
                "step": "reverse_traversal",
                "detail": retrieval.connected_error,
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

        arbiter.join(timeout=60)
        arbitration = adjudged.get("result") or trust.Arbitration(claims=(), conflicts=())
        for conflict in arbitration.conflicts:
            yield {
                "type": "conflict_found",
                "subject": conflict.subject,
                "predicate": conflict.predicate,
                "values": [
                    {"value": c.object_value, "source": c.source,
                     "date": c.asserted_at, "cite": c.doc_id, "trust": c.trust}
                    for c in (conflict.winner, *conflict.losers)
                ],
            }
            yield {
                "type": "winner_chosen",
                "subject": conflict.subject,
                "predicate": conflict.predicate,
                # `decided` is false when the margin was too thin to justify a
                # choice. That is a result, not a failure, and the interface
                # must not draw it as a verdict.
                "decided": conflict.decided,
                "value": conflict.winner.object_value if conflict.decided else None,
                "source": conflict.winner.source,
                "why": conflict.rationale,
                "rejected": [
                    {"value": loser.object_value, "source": loser.source,
                     "status": loser.status}
                    for loser in conflict.losers
                ],
            }

        # Now compose the answer, with graph evidence included.
        written: queue.Queue = queue.Queue()

        def compose() -> None:
            try:
                for part in answer.write_streaming(question, docs, people, paths=paths,
                                                connected=retrieval.connected_entities,
                                                facets=facets, arbitration=arbitration):
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
            "claims": len(arbitration.claims),
            "conflicts": len(arbitration.conflicts),
            "ms": round((time.time() - t0) * 1000),
        }

    def ask(self, question: str, limit: int = 20) -> Answer:
        t0 = time.time()
        events: list[Event] = []

        retrieval = self.retrieve(question, limit)
        docs = self._answer_documents(retrieval, limit)
        facets = self._facets_for(docs[: answer.MAX_DOCS])
        arbitration = self.adjudicate(question, docs[: answer.MAX_DOCS])
        events.append(
            Event(
                "recall",
                {
                    "documents": len(docs),
                    "terms": len(self.recall.selective_terms(question)),
                },
            )
        )
        if retrieval.containers:
            events.append(
                Event(
                    "containers",
                    {
                        "named": len(retrieval.containers),
                        "entrance": retrieval.container_entrance,
                        "documents": len(retrieval.container_docs),
                        "available": retrieval.container_error is None,
                    },
                )
            )
        if arbitration.claims:
            events.append(
                Event(
                    "claims",
                    {
                        "extracted": len(arbitration.claims),
                        "conflicts": len(arbitration.conflicts),
                        "decided": sum(
                            1 for c in arbitration.conflicts if c.decided
                        ),
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
            # Worth recording, never worth refusing over. This used to skip
            # synthesis outright, on the reasoning that we should not discuss
            # material whose people we cannot name -- but most questions are
            # not about people at all, and the ones that are usually have the
            # person as the *answer*: "who authored the SLO throttler PR" names
            # nobody, resolves nobody, and was answered with an empty string
            # while the correct document sat in the context. `stream` never had
            # this gate, so the interface answered questions the graded path
            # silently declined. Rule 2 of the prompt already handles evidence
            # that does not contain the answer, and it handles it per question
            # rather than for a whole class of them.
            events.append(Event("no_identity", {"documents": len(docs)}))

        text = ""
        cited: list[int] = []
        try:
            written = answer.write(question, docs, people, paths=paths,
                          connected=retrieval.connected_entities,
                          facets=facets, arbitration=arbitration)
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
