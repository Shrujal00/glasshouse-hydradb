"""HydraDB open-source engine — the ontology and reasoning layer.

The cloud half (`cloud.py`) answers "which documents might bear on this?".
This half answers "what is true, per whom, as of when?" — and, for entity
resolution specifically, "why do we believe these two names are one person?".

What the engine is and is not used for
--------------------------------------
Its OpenCypher subset is narrow in ways that shape the design rather than
merely annoy it (all confirmed by probing in Session 0):

- `CREATE` cannot be followed by another clause, and `WITH` is pass-through
  only. So iterative algorithms cannot be written in Cypher at all.
- `RETURN` yields `<binding>.<property>` or `count(*)`, never whole nodes.
- `WHERE` has no `IN`/`CONTAINS`/`IS NULL`; set reads go through `UNWIND`.
- Node ids must be non-negative integers, so string keys need a hash plus a
  registry — see `node_id`.
- Property values are scalars only, which is why aliases are nodes rather than
  a list property.

The consequence is a deliberate split, not a workaround: **pair scoring and
clustering run in Python; the graph holds the result and does the traversal.**
That is the operation the engine is actually exceptional at — `algo.SPpaths`
returns whole paths with every node and relationship property inline, which is
simultaneously the multi-hop reasoning primitive the benchmark needs and the
exact payload the reasoning canvas animates.

So the resolution graph written here is not a report of a decision taken
elsewhere. It is the decision, in the form that makes it answerable: every
alias, every entity, every accepted merge with its evidence, and every merge
that was *refused* and why. "Who is @jae, and why isn't it Jordan?" is a path
query, not a grep.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .config import get, require

# The engine rejects negative ids, so the sign bit is masked off. 63 bits of
# blake2b over a stable string key keeps collisions far below the ~10^6 nodes
# this graph will hold, and makes ids reproducible across runs and machines -
# a rebuild must not renumber the graph.
_ID_BITS = (1 << 63) - 1


def node_id(key: str) -> int:
    """Deterministic non-negative int64 id for a string key."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") & _ID_BITS


class GraphError(RuntimeError):
    pass


_WRITES = re.compile(r"\b(MERGE|CREATE|SET|DELETE|REMOVE)\b")


def write_key(cypher: str, parameters: dict[str, Any] | None) -> str:
    """Idempotency key for a write: a digest of the statement and its payload.

    Unique per distinct write, identical for a replay of the same one, which is
    exactly the contract the engine's importer wants — see `query`.
    """
    body = json.dumps(parameters or {}, sort_keys=True, default=str)
    return hashlib.blake2b(f"{cypher}\x00{body}".encode(), digest_size=16).hexdigest()


@dataclass(slots=True)
class Row:
    """One result row, already unwrapped from the engine's typed encoding."""

    values: dict[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.values[name]


@dataclass(frozen=True, slots=True)
class GraphCandidate:
    """A document reached from a resolved entity, not evidence of its contents."""

    doc_id: str
    seed_eids: tuple[str, ...]
    path: dict[str, Any]
    hops: int
    reason: str


@dataclass(slots=True)
class SurfaceMatch:
    """Who one written form denotes, read straight off the traversal path.

    `entities` is the ambiguity guard and belongs to the *surface*, not to the
    person: a form that reaches two people has not named either of them. It is
    counted over the whole ontology when the graph is loaded, because the
    engine answers anchored traversals and rejects the scan that would count it
    at query time.
    """

    text: str
    kinds: tuple[str, ...]
    entities: int
    given_name_forms: int
    eid: str
    name: str
    node: int
    confidence: float
    alias_count: int


@dataclass(slots=True)
class EntityCandidate:
    """A person connected to evidence, not proof that they own or decided it."""

    eid: str
    name: str
    node: int
    doc_ids: tuple[str, ...]
    reason: str

    @property
    def documents(self) -> int:
        return len(self.doc_ids)


@dataclass(frozen=True, slots=True)
class ClaimNode:
    """One assertion, as the graph stores it.

    Distinct from `claims.Claim`, which is what the extractor produced for one
    question in one process. This is the same fact after it has been arbitrated,
    written, and read back by whoever asks next -- which is the entire point of
    putting it in the graph rather than in a cache.
    """

    claim_id: str
    scope: str
    subject: str
    predicate: str
    object_value: str
    doc_id: str
    title: str
    source: str
    asserted_at: str
    trust: float
    status: str
    rationale: str
    gen: str
    node: int

    @property
    def cite(self) -> str:
        where = self.source or "unknown source"
        if self.asserted_at:
            where += f", {self.asserted_at}"
        return f"{self.title or self.doc_id} ({where})"


@dataclass(frozen=True, slots=True)
class DisagreementNode:
    """One group of claims about the same thing that do not agree.

    Every counting field on here was computed by the loader. The engine has no
    aggregate beyond `count(*)` and no `GROUP BY` at all, so a disagreement
    that could only be described by aggregating its claims could not be ranked,
    filtered or listed -- it would be visible only to someone who already knew
    which one to ask for.
    """

    key: str
    scope: str
    subject: str
    predicate: str
    sides: int
    claims: int
    documents: int
    sources: tuple[str, ...]
    trust_gap: float
    decided: bool
    winner_value: str
    winner_source: str
    winner_trust: float
    winner_claim_id: str
    runner_value: str
    runner_source: str
    runner_trust: float
    runner_claim_id: str
    rationale: str
    first_asserted: str
    last_asserted: str
    entity_eid: str
    entity_name: str
    weight: float
    gen: str
    node: int


# `RETURN` yields `<binding>.<property>`, never a whole node, so every read has
# to name its fields. Kept as one string so the shape a caller gets back cannot
# drift between the four statements that produce it.
_CLAIM_FIELDS = (
    "c.claim_id AS claim_id, c.scope AS scope, c.subject AS subject, "
    "c.predicate AS predicate, c.object_value AS object_value, c.doc_id AS doc_id, "
    "c.title AS title, c.source AS source, c.asserted_at AS asserted_at, "
    "c.trust AS trust, c.status AS status, c.rationale AS rationale, "
    "c.gen AS gen, c.id AS node"
)

_DISAGREEMENT_FIELDS = (
    "d.key AS key, d.scope AS scope, d.subject AS subject, d.predicate AS predicate, "
    "d.sides AS sides, d.claims AS claims, d.documents AS documents, "
    "d.sources AS sources, d.trust_gap AS trust_gap, d.decided AS decided, "
    "d.winner_value AS winner_value, d.winner_source AS winner_source, "
    "d.winner_trust AS winner_trust, d.winner_claim_id AS winner_claim_id, "
    "d.runner_value AS runner_value, d.runner_source AS runner_source, "
    "d.runner_trust AS runner_trust, d.runner_claim_id AS runner_claim_id, "
    "d.rationale AS rationale, d.first_asserted AS first_asserted, "
    "d.last_asserted AS last_asserted, d.entity_eid AS entity_eid, "
    "d.entity_name AS entity_name, d.weight AS weight, d.gen AS gen, d.id AS node"
)


def _literal(text: str) -> str:
    """A value safe to inline into a statement.

    Filters are inlined rather than parameterised because the engine accepts
    `$rows` in an `UNWIND` and nowhere useful in a `WHERE`. Quotes and
    backslashes are stripped rather than escaped: every caller of this is
    filtering on a predicate name or a ticket key, neither of which contains
    either, and a filter that silently matches nothing is a better failure than
    one that changes the statement.
    """
    return re.sub(r"[\\'\"\x00-\x1f]", "", str(text))[:120]


def _claim(v: dict[str, Any]) -> ClaimNode:
    return ClaimNode(
        claim_id=str(v.get("claim_id") or ""),
        scope=str(v.get("scope") or ""),
        subject=str(v.get("subject") or ""),
        predicate=str(v.get("predicate") or ""),
        object_value=str(v.get("object_value") or ""),
        doc_id=str(v.get("doc_id") or ""),
        title=str(v.get("title") or ""),
        source=str(v.get("source") or ""),
        asserted_at=str(v.get("asserted_at") or ""),
        trust=float(v.get("trust") or 0.0),
        status=str(v.get("status") or ""),
        rationale=str(v.get("rationale") or ""),
        gen=str(v.get("gen") or ""),
        node=int(v.get("node") or 0),
    )


def _disagreement(v: dict[str, Any]) -> DisagreementNode:
    return DisagreementNode(
        key=str(v.get("key") or ""),
        scope=str(v.get("scope") or ""),
        subject=str(v.get("subject") or ""),
        predicate=str(v.get("predicate") or ""),
        sides=int(v.get("sides") or 0),
        claims=int(v.get("claims") or 0),
        documents=int(v.get("documents") or 0),
        sources=tuple(s for s in str(v.get("sources") or "").split("|") if s),
        trust_gap=float(v.get("trust_gap") or 0.0),
        # Booleans are stored as 0/1: the engine takes a boolean property but
        # returns it inconsistently enough that comparing to an integer is the
        # only form that behaves the same on a write and on a read.
        decided=bool(int(v.get("decided") or 0)),
        winner_value=str(v.get("winner_value") or ""),
        winner_source=str(v.get("winner_source") or ""),
        winner_trust=float(v.get("winner_trust") or 0.0),
        winner_claim_id=str(v.get("winner_claim_id") or ""),
        runner_value=str(v.get("runner_value") or ""),
        runner_source=str(v.get("runner_source") or ""),
        runner_trust=float(v.get("runner_trust") or 0.0),
        runner_claim_id=str(v.get("runner_claim_id") or ""),
        rationale=str(v.get("rationale") or ""),
        first_asserted=str(v.get("first_asserted") or ""),
        last_asserted=str(v.get("last_asserted") or ""),
        entity_eid=str(v.get("entity_eid") or ""),
        entity_name=str(v.get("entity_name") or ""),
        weight=float(v.get("weight") or 0.0),
        gen=str(v.get("gen") or ""),
        node=int(v.get("node") or 0),
    )


def _unwrap(cell: Any) -> Any:
    """Values arrive as {"type": ..., "value": ...}; paths nest the same shape."""
    if not isinstance(cell, dict) or "value" not in cell:
        return cell
    kind, value = cell.get("type"), cell["value"]
    if kind == "path" and isinstance(value, dict):
        return {
            "nodes": [_unwrap_props(n) for n in value.get("nodes", [])],
            "relationships": [_unwrap_props(r) for r in value.get("relationships", [])],
        }
    return value


# Inside a path, properties are tagged by type name rather than by the
# {"type","value"} shape the top level uses: {"nm": {"String": "mchen"}}.
_TAGS = frozenset(("String", "Integer", "Float", "Boolean", "Bool", "Null"))


def _unwrap_props(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    if len(item) == 1 and next(iter(item)) in _TAGS:
        return next(iter(item.values()))
    return {
        k: ({p: _unwrap_props(q) for p, q in v.items()} if k == "properties" and isinstance(v, dict) else _unwrap(v))
        for k, v in item.items()
    }


class GraphEngine:
    """Thin HTTP client for the engine's Cypher endpoint."""

    def __init__(self, base: str | None = None, token: str | None = None) -> None:
        self.base = (base or get("HYDRA_HTTP_URI", "http://127.0.0.1:8443")).rstrip("/")
        self.token = token or require("HYDRA_LOCAL_TOKEN")
        self.namespace = get("HYDRA_GRAPH_NAMESPACE", "default")
        self.graph_id = get("HYDRA_GRAPH_ID", "default")
        self.cell_id = get("HYDRA_GRAPH_CELL_ID", "cell-0")

    def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        *,
        strong: bool = False,
        timeout: float = 120.0,
    ) -> list[Row]:
        """Run one statement. Only one - the engine has no multi-statement form.

        `strong` is required to read your own writes: the default `causal`
        consistency returns empty rows immediately after a committed write,
        which reads as data loss rather than as staleness.
        """
        payload: dict[str, Any] = {"query": cypher, "cell_id": self.cell_id}
        if parameters:
            payload["parameters"] = parameters
        if strong:
            payload["consistency"] = "strong"
        # Every import the engine performs is deduplicated against an
        # idempotency key, and when the payload does not name one the engine
        # invents it from a per-process request counter:
        # `http-query-104.unwind-relationship-merge`. That counter restarts
        # with the container, so after a restart each write collides with
        # whatever the previous run stored under the same number and comes back
        # as `500 internal query execution error` — measured on the loaded
        # graph as 3 of 4 relationship MERGEs failing, non-deterministically,
        # with the real reason only visible in `docker logs`. Keying on the
        # payload instead makes the key unique per statement and identical for
        # a replay, which is what makes the retry below idempotent rather than
        # merely lucky.
        if _WRITES.search(cypher):
            payload["query_id"] = write_key(cypher, parameters)

        request = urllib.request.Request(
            f"{self.base}/v1/graphs/{self.graph_id}/query",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-Graph-Namespace": self.namespace,
                "Content-Type": "application/json",
            },
        )
        # The engine occasionally answers a perfectly valid statement with
        # `internal query execution error`, and the same statement succeeds a
        # moment later. Retry those rather than surfacing a transient hiccup as
        # a failed answer; anything the parser rejects is deterministic and is
        # raised on the first attempt.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()[:500]
                transient = exc.code >= 500 or "internal" in detail
                if not transient or attempt == 2:
                    raise GraphError(f"{exc.code} on `{cypher[:90]}`: {detail}") from None
                time.sleep(0.25 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise GraphError(f"unreachable on `{cypher[:90]}`: {exc}") from None
                time.sleep(0.25 * (attempt + 1))

        rows = body.get("rows") or body.get("data") or []
        columns = body.get("columns") or []
        out: list[Row] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(Row({k: _unwrap(v) for k, v in row.items()}))
            else:
                out.append(Row({c: _unwrap(v) for c, v in zip(columns, row)}))
        return out

    # --- lifecycle ----------------------------------------------------------

    def wait_until_ready(self, timeout: float = 90.0) -> bool:
        """Poll until the engine answers a query.

        Readiness is checked here rather than by a container healthcheck: the
        image carries no curl or wget, so any in-container probe fails forever
        while the engine serves perfectly well.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.query("MATCH (n:Probe) RETURN count(*)")
                return True
            except Exception:
                time.sleep(2.0)
        return False

    def count(self, label: str) -> int:
        rows = self.query(f"MATCH (n:{label}) RETURN count(*)", strong=True)
        return int(next(iter(rows[0].values.values()))) if rows else 0

    # --- writes -------------------------------------------------------------

    def upsert_nodes(self, label: str, rows: Sequence[dict[str, Any]], properties: Sequence[str]) -> None:
        """MERGE by integer id, then SET the properties.

        `MERGE` here takes no `ON CREATE`/`ON MATCH` and will not accept extra
        properties folded into its pattern, so the id match and the property
        write are two clauses of the same statement.
        """
        if not rows:
            return
        sets = ", ".join(f"n.{p} = row.{p}" for p in properties)
        self.query(
            f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:{label}, {sets}",
            {"rows": list(rows)},
        )

    def create_edges(
        self,
        rel: str,
        rows: Sequence[dict[str, Any]],
        properties: Sequence[str],
        src_label: str,
        dst_label: str | None = None,
    ) -> None:
        """CREATE one relationship type between already-existing nodes.

        Three parser requirements, all discovered the hard way and all in
        `cypher-compat.md`:

        - Both endpoints must carry a label. A bare `{id: ...}` match is
          rejected with "endpoints require exactly one label".
        - The relationship itself needs an integer `id` property, exactly as
          the endpoints do. Omitting it fails with "CREATE properties require
          id: row.<field>".
        - One relationship type per pattern, one hop, directed, and nothing
          may follow the CREATE.

        The edge id is derived from the endpoints and type when the caller
        does not supply one, so a replayed batch writes the same ids rather
        than a second parallel edge.
        """
        if not rows:
            return
        rows = [
            r if "id" in r else {**r, "id": node_id(f"{r['src']}-{rel}->{r['dst']}")}
            for r in rows
        ]
        props = ", ".join(f"{p}: row.{p}" for p in ("id", *properties))
        self.query(
            f"UNWIND $rows AS row "
            f"MATCH (s:{src_label} {{id: row.src}}), (d:{dst_label or src_label} {{id: row.dst}}) "
            f"CREATE (s)-[:{rel} {{{props}}}]->(d)",
            {"rows": list(rows)},
        )

    def merge_edges(
        self,
        rel: str,
        rows: Sequence[dict[str, Any]],
        properties: Sequence[str],
        src_label: str,
        dst_label: str | None = None,
    ) -> None:
        """Idempotent edge write: MERGE on the edge id, then SET its properties.

        Preferred over `create_edges` for anything that may run twice. Loading
        the ontology used to clear the graph first, but `DETACH DELETE` over
        every node exceeds the engine's 30-second query ceiling well before the
        corpus is fully loaded. Deterministic ids plus MERGE make a reload
        overwrite in place instead, which is both faster and safe to interrupt.
        """
        if not rows:
            return
        rows = [
            r if "id" in r else {**r, "id": node_id(f"{r['src']}-{rel}->{r['dst']}")}
            for r in rows
        ]
        # An edge that carries nothing but its endpoints is a real thing to
        # want -- `EVIDENCED_BY` says everything by existing -- and the parser
        # rejects a `SET` with no assignments after it, so the clause is
        # omitted rather than padded with a placeholder property.
        sets = ", ".join(f"r.{p} = row.{p}" for p in properties)
        self.query(
            f"UNWIND $rows AS row "
            f"MATCH (s:{src_label} {{id: row.src}}), (d:{dst_label or src_label} {{id: row.dst}}) "
            f"MERGE (s)-[r:{rel} {{id: row.id}}]->(d)" + (f" SET {sets}" if sets else ""),
            {"rows": list(rows)},
        )

    def wipe(self, labels: Iterable[str]) -> None:
        """Drop every node of the given labels.

        Only viable on a small graph: the engine caps a query at 30 seconds and
        an unbounded `DETACH DELETE` over a loaded ontology will exceed it. To
        start genuinely clean, recreate the container volume instead
        (`docker compose down -v && docker compose up -d`).
        """
        for label in labels:
            self.query(f"MATCH (n:{label}) DETACH DELETE n")

    # --- traversal ----------------------------------------------------------

    def paths(
        self,
        source: int,
        target: int,
        rel_types: Sequence[str],
        max_len: int = 3,
        path_count: int = 5,
    ) -> list[dict[str, Any]]:
        """Bounded multi-hop paths, whole, with properties inline.

        This is the engine earning its place: one call returns the nodes, the
        relationships and every property along each path, which is both the
        answer to a multi-hop question and the frames the canvas animates.
        """
        rels = ", ".join(f"'{r}'" for r in rel_types)
        rows = self.query(
            f"CALL algo.SPpaths({{sourceNode: {source}, targetNode: {target}, "
            f"relTypes: [{rels}], relDirection: 'both', maxLen: {max_len}, "
            f"pathCount: {path_count}}}) YIELD path, pathCost RETURN path, pathCost",
            strong=True,
        )
        return [{"path": r.values.get("path"), "cost": r.values.get("pathCost")} for r in rows]

    def entities_for_documents(
        self, doc_ids: Sequence[str], limit: int = 200, documents: int = 8
    ) -> list["EntityCandidate"]:
        """Who the graph connects to these documents, most-corroborated first.

        `documents_for_entities` is the wrong way round for the questions that
        matter. "Who owns the audit-log shipper sidecar?" names no one -- the
        person is the answer -- so seeding from the question yields nothing.
        Retrieval already found the documents about the component; reading the
        same `MENTIONED_IN` edge inwards turns them into the people attached to
        that evidence.

        Ranking is by how many of the retrieved documents each person is
        connected to. Appearing across four of six documents about a component
        is a materially different signal from appearing in one, and it is a
        count over edges rather than over words, which is the part keyword
        search cannot do. It remains co-occurrence: connected is not owner.
        """
        pages = [d for d in dict.fromkeys(doc_ids) if d][:documents]
        if not pages or limit <= 0:
            return []
        found: dict[str, EntityCandidate] = {}
        order: list[str] = []
        for doc_id in pages:
            rows = self.query(
                f"CALL algo.SSpaths({{sourceNode: {node_id('doc:' + doc_id)}, "
                f"relTypes: ['MENTIONED_IN'], relDirection: 'incoming', "
                f"maxLen: 1, pathCount: {limit}}}) YIELD path RETURN path",
                strong=True,
            )
            for row in rows:
                path = row.values.get("path") or {}
                for node in path.get("nodes", []):
                    if not isinstance(node, dict):
                        continue
                    props = node.get("properties") or {}
                    eid = str(props.get("eid") or "")
                    # The Document end of every path carries no eid, which is
                    # what keeps it out of the tally.
                    if not eid:
                        continue
                    entity = found.get(eid)
                    if entity is None:
                        entity = found[eid] = EntityCandidate(
                            eid=eid,
                            name=str(props.get("canonical_name") or eid),
                            node=int(node.get("id") or -1),
                            doc_ids=(),
                            reason="Entity-[:MENTIONED_IN]->Document, read inwards",
                        )
                        order.append(eid)
                    if doc_id not in entity.doc_ids:
                        entity.doc_ids = entity.doc_ids + (doc_id,)
        return sorted(
            (found[eid] for eid in order),
            key=lambda e: (-e.documents, order.index(e.eid)),
        )

    def denoted_by(self, text: str, limit: int = 4) -> list[SurfaceMatch]:
        """Every person a written form could mean. One anchored hop.

        This is the identity half of the brief -- deciding that `sam`,
        `@soham` and `S. Ratnaparkhi` are one person -- answered by traversal
        rather than by a lookup table. `node_id` turns the word into an anchor,
        so a word taken out of a question costs one hop and no scan.

        Returns every match rather than only the unambiguous one, because the
        caller is entitled to know the difference between a form nobody uses
        and a form eight people share; both are reasons not to expand it, and
        only one of them is worth telling the reader about.
        """
        word = (text or "").strip().lower()
        if not word:
            return []
        try:
            # An anchored `MATCH` rather than `algo.SSpaths`: both cost about
            # 0.09s, but this one returns the fields directly instead of a
            # path to be walked, and the engine accepts it because the pattern
            # is pinned to a literal node id. It will not accept the same
            # statement under `UNWIND`, so there is no batched form -- every
            # word costs a round trip, which is what bounds how many words a
            # question is allowed to ask about.
            rows = self.query(
                f"MATCH (s:Surface {{id: {node_id('surface:' + word)}}})-[:DENOTES]->(e:Entity) "
                "RETURN s.text AS text, s.kinds AS kinds, s.entities AS entities, "
                "s.given_name_forms AS given, e.eid AS eid, e.canonical_name AS name, "
                "e.confidence AS confidence, e.alias_count AS alias_count, e.id AS node "
                f"LIMIT {int(limit)}",
                strong=True,
            )
        except GraphError:
            # An absent anchor is not an error, and neither is a graph that has
            # not had `load_surface_graph.py` run against it. The caller falls
            # back; taking the answer down would be a worse trade.
            return []
        out: list[SurfaceMatch] = []
        for row in rows:
            v = row.values
            if not v.get("eid"):
                continue
            out.append(
                SurfaceMatch(
                    text=str(v.get("text") or word),
                    kinds=tuple(k for k in str(v.get("kinds") or "").split("|") if k),
                    entities=int(v.get("entities") or len(rows)),
                    given_name_forms=int(v.get("given") or 0),
                    eid=str(v["eid"]),
                    name=str(v.get("name") or ""),
                    node=int(v.get("node") or -1),
                    confidence=float(v.get("confidence") or 0.0),
                    alias_count=int(v.get("alias_count") or 1),
                )
            )
        return out

    def surfaces_of(self, entity_node: int, limit: int = 40) -> list[tuple[str, str]]:
        """Every `(form, kind)` that denotes this person — the same edge, inwards.

        The forward direction answers "who is this word"; this one answers
        "what else is this person called", which is what retrieval expands a
        query with, and what personhood is judged on: an entity carrying a
        `name` alongside a second spelling is a person, while one carrying a
        single `handle` is a channel tag the parser noticed once. Both answers
        come from this one traversal rather than from four counting queries.
        """
        try:
            rows = self.query(
                f"CALL algo.SSpaths({{sourceNode: {int(entity_node)}, "
                f"relTypes: ['DENOTES'], relDirection: 'incoming', "
                f"maxLen: 1, pathCount: {int(limit)}}}) YIELD path RETURN path",
                strong=True,
            )
        except GraphError:
            return []
        found: dict[str, tuple[str, ...]] = {}
        for row in rows:
            for node in (row.values.get("path") or {}).get("nodes", []):
                if isinstance(node, dict):
                    props = node.get("properties") or {}
                    text = props.get("text")
                    if text:
                        found.setdefault(
                            str(text),
                            tuple(k for k in str(props.get("kinds") or "").split("|") if k),
                        )
        # One pair per role: a form written as both a handle and a name is two
        # pieces of evidence about the person, not one.
        return [(text, kind) for text, kinds in found.items() for kind in (kinds or ("",))]

    def documents_in_containers(
        self, containers: Sequence[tuple[str, int]], limit: int = 200
    ) -> list[GraphCandidate]:
        """Documents held by the containers a question named, one hop inwards.

        The third entrance. `documents_for_entities` needs the question to name
        a person and `entities_for_documents` needs retrieval to have already
        found the right documents; a question like "in the internal customer
        success and support knowledge space" names neither, it names a *place*.
        Folders and channels are that place, and they are nodes here, so the
        scope is a single anchored hop rather than a keyword guess.

        Containers arrive as `(container_key, node_id)` pairs, the same shape
        `documents_for_entities` takes, because the caller holds the local
        facet table and already knows both halves; the node id is always
        `node_id(f"container:{key}")`.

        Fails soft on purpose. The container half of the graph may not be
        loaded yet -- the load is long and the engine has an ingest ceiling --
        and while it is missing the local `FacetStore` serves the same scope.
        An absent anchor returns no paths rather than an error, and a rejected
        query is skipped, so this returns `[]` instead of taking the answer
        down with it. The trace names which entrance actually opened, so an
        empty return here is visible rather than papered over.
        """
        if not containers or limit <= 0:
            return []
        rows_by_container: list[tuple[str, int, list[tuple[str, int]]]] = []
        for key, container_id in containers:
            try:
                # Anchored at the container exactly as the entity path is
                # anchored at the person: read the anchor's own adjacency with
                # `algo.SSpaths` rather than expanding a labeled `MATCH`, which
                # scans the whole edge set and blows the engine's 30s cap.
                rows = self.query(
                    f"CALL algo.SSpaths({{sourceNode: {int(container_id)}, "
                    f"relTypes: ['IN_CONTAINER'], relDirection: 'incoming', "
                    f"maxLen: 1, pathCount: {limit}}}) YIELD path RETURN path",
                    strong=True,
                )
            except GraphError:
                continue
            reached: list[tuple[str, int]] = []
            seen_docs: set[str] = set()
            for row in rows:
                path = row.values.get("path") or {}
                for node in path.get("nodes", []):
                    if not isinstance(node, dict):
                        continue
                    doc_id = str((node.get("properties") or {}).get("doc_id") or "")
                    # The Container node rides along in every path and carries
                    # a key rather than a doc_id, which is what excludes it.
                    if doc_id and doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        reached.append((doc_id, int(node.get("id") or -1)))
            rows_by_container.append((key, int(container_id), reached))

        # Round-robin, as with entity seeds: #incidents holds 28,999 documents
        # and the median container holds one, so taking containers in turn
        # keeps the huge one from consuming the entire scope.
        by_doc: dict[str, list[tuple[str, int, int]]] = {}
        for rank in range(limit):
            for key, container_id, reached in rows_by_container:
                if rank < len(reached):
                    doc_id, document_node = reached[rank]
                    by_doc.setdefault(doc_id, []).append((key, container_id, document_node))

        candidates: list[GraphCandidate] = []
        for doc_id, reached in list(by_doc.items())[:limit]:
            paths = []
            seen_keys: set[str] = set()
            for key, container_id, document_node in reached:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                paths.append(
                    {
                        "seed_eid": key,
                        "nodes": [
                            {
                                "id": document_node,
                                "labels": ["Document"],
                                "properties": {"doc_id": doc_id},
                            },
                            {"id": container_id, "labels": ["Container"], "properties": {"key": key}},
                        ],
                        "relationships": [
                            {
                                "edge_type": "IN_CONTAINER",
                                "src": document_node,
                                "dst": container_id,
                            }
                        ],
                    }
                )
            candidates.append(
                GraphCandidate(
                    doc_id=doc_id,
                    # The seed is a container, not a person. Callers render this
                    # into the trace, so it carries the container key and the
                    # reason says container, never "reached from an entity".
                    seed_eids=tuple(sorted(seen_keys)),
                    path={"paths": paths},
                    hops=1,
                    reason="direct Document-[:IN_CONTAINER]->Container membership",
                )
            )
        return candidates

    def documents_for_entities(
        self, seeds: Sequence[tuple[str, int]], limit: int = 200
    ) -> list[GraphCandidate]:
        """Return direct Entity-to-Document neighbors for resolved query entities.

        This anchored query intentionally avoids global Document or relationship
        scans, which exceed the engine's admission limits on the loaded corpus.
        """
        if not seeds or limit <= 0:
            return []
        rows_by_seed: list[tuple[str, int, list[tuple[str, int]]]] = []
        by_doc: dict[str, list[tuple[str, int, int]]] = {}
        for eid, seed_id in seeds:
            # A labeled MATCH expansion is the obvious way to write this and is
            # unusable here: on the loaded corpus it scans the whole
            # MENTIONED_IN edge set, taking 15-30s per seed and often exceeding
            # the engine's 30s cap, even with LIMIT 1 and even for an entity
            # with a single alias. Anchoring the same hop with the native
            # single-source path procedure reads the anchor's adjacency instead
            # and returns 200 paths in well under a second. The engine also
            # rejects labeled node patterns after UNWIND, so seeds stay one
            # bounded query each rather than one batched set query.
            rows = self.query(
                f"CALL algo.SSpaths({{sourceNode: {int(seed_id)}, "
                f"relTypes: ['MENTIONED_IN'], relDirection: 'outgoing', "
                f"maxLen: 1, pathCount: {limit}}}) YIELD path RETURN path",
                strong=True,
            )
            reached: list[tuple[str, int]] = []
            seen_docs: set[str] = set()
            for row in rows:
                path = row.values.get("path") or {}
                for node in path.get("nodes", []):
                    if not isinstance(node, dict):
                        continue
                    doc_id = str((node.get("properties") or {}).get("doc_id") or "")
                    # The source Entity node rides along in every path and
                    # carries no doc_id, which is what excludes it here.
                    if doc_id and doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        reached.append((doc_id, int(node.get("id") or -1)))
            rows_by_seed.append((eid, seed_id, reached))

        # Round-robin keeps one high-degree entity from consuming the entire
        # scope and still lets shared documents accumulate all seed paths.
        for rank in range(limit):
            for eid, seed_id, reached in rows_by_seed:
                if rank < len(reached):
                    doc_id, document_node = reached[rank]
                    by_doc.setdefault(doc_id, []).append((eid, seed_id, document_node))

        candidates: list[GraphCandidate] = []
        for doc_id, reached in list(by_doc.items())[:limit]:
            seed_eids = tuple(sorted({eid for eid, _, _ in reached}))
            paths = []
            seen_seeds: set[str] = set()
            for eid, seed_id, document_node in reached:
                if eid in seen_seeds:
                    continue
                seen_seeds.add(eid)
                paths.append(
                    {
                        "seed_eid": eid,
                        "nodes": [
                            {"id": seed_id, "labels": ["Entity"]},
                            {
                                "id": document_node,
                                "labels": ["Document"],
                                "properties": {"doc_id": doc_id},
                            },
                        ],
                        "relationships": [
                            {
                                "edge_type": "MENTIONED_IN",
                                "src": seed_id,
                                "dst": document_node,
                            }
                        ],
                    }
                )
            candidates.append(
                GraphCandidate(
                    doc_id=doc_id,
                    seed_eids=seed_eids,
                    path={"paths": paths},
                    hops=1,
                    reason="direct Entity-[:MENTIONED_IN]->Document reachability",
                )
            )
        return candidates

    # --- the contradiction graph --------------------------------------------

    def disagreements(
        self,
        limit: int = 40,
        *,
        undecided_only: bool = False,
        predicate: str = "",
        scope: str = "",
        gen: str = "",
    ) -> list["DisagreementNode"]:
        """Everything the organisation contradicts itself about, worst first.

        This is a label scan, and a label scan is exactly what the engine
        rejects on `Entity` and `Document`. It is allowed here because
        `Disagreement` is a few hundred nodes rather than a few hundred
        thousand: the same statement over `Document` returns
        `429 resource_exhausted`, and over `Disagreement` it returns in about a
        twentieth of a second. Size is the whole difference, so the loader's
        job is to keep this label small enough to stay scannable.

        The ranking properties -- how many documents back each side, how wide
        the trust gap is, how many sources are involved -- are written by the
        loader rather than aggregated here. `count(...) AS n` is rejected
        outright by the parser, so anything a `GROUP BY` would have produced
        has to exist as a property before the question is asked.

        `gen` filters to one load. Nothing in this graph can be deleted --
        `DETACH DELETE` is refused by admission control even for a single
        anchored node with two edges, because deleting a vertex scans its
        edges and these are wired to `Document` -- so a reload that changes
        how claims are extracted leaves the previous run's nodes behind
        forever. Stamping each load and reading one stamp is the only way to
        show a current map rather than an accumulation of every map ever
        built.
        """
        where = []
        if gen:
            where.append(f"d.gen = '{_literal(gen)}'")
        if undecided_only:
            where.append("d.decided = 0")
        if predicate:
            where.append(f"d.predicate = '{_literal(predicate)}'")
        if scope:
            where.append(f"d.scope = '{_literal(scope)}'")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        try:
            rows = self.query(
                "MATCH (d:Disagreement)" + clause + " RETURN " + _DISAGREEMENT_FIELDS
                + f" ORDER BY d.weight DESC LIMIT {int(limit)}",
                strong=True,
            )
        except GraphError:
            return []
        return [_disagreement(row.values) for row in rows]

    def disagreement(
        self, key: str, *, gen: str = ""
    ) -> tuple["DisagreementNode | None", list["ClaimNode"]]:
        """One disagreement and every claim on every side of it.

        Two anchored round trips rather than one statement: the summary and the
        claims are different shapes, and the engine returns properties rather
        than nodes, so a single query would have to repeat every summary field
        on every claim row.

        The claims are filtered by generation for the same reason the listing
        is, and it matters more here. A disagreement keeps its node id across
        reloads -- it is a digest of scope, subject and predicate, none of
        which change -- while a claim's id is a digest of its *value*, which
        does. So a reload leaves the old claims wired to the current
        disagreement by `OVER` edges that cannot be deleted, and the panel
        shows a claim marked `accepted` inside a disagreement the system says
        it refused to decide. Two states of one arbitration on screen at once
        is worse than showing nothing.
        """
        anchor = node_id(f"disagreement:{key}")
        try:
            head = self.query(
                f"MATCH (d:Disagreement {{id: {anchor}}}) RETURN " + _DISAGREEMENT_FIELDS,
                strong=True,
            )
            if not head:
                return None, []
            rows = self.query(
                f"MATCH (d:Disagreement {{id: {anchor}}})-[:OVER]->(c:Claim)"
                + (f" WHERE c.gen = '{_literal(gen)}'" if gen else "")
                + " RETURN "
                + _CLAIM_FIELDS
                + " ORDER BY c.trust DESC",
                strong=True,
            )
        except GraphError:
            return None, []
        return _disagreement(head[0].values), [_claim(row.values) for row in rows]

    def claim_history(self, claim_id: str, depth: int = 6) -> list[dict[str, Any]]:
        """What this value used to be, and what corrected it.

        `SUPERSEDES` points from the claim that won to the claim it replaced,
        so walking it outwards from the current value is walking backwards in
        time. `algo.SSpaths` returns the whole chain in one call with every
        property inline -- the alternative is one anchored hop per link at
        ~0.09s each, and the chain is the answer rather than the last node on
        it.
        """
        try:
            rows = self.query(
                f"CALL algo.SSpaths({{sourceNode: {node_id('claim:' + claim_id)}, "
                f"relTypes: ['SUPERSEDES'], relDirection: 'outgoing', "
                f"maxLen: {int(depth)}, pathCount: {int(depth) * 4}}}) YIELD path RETURN path",
                strong=True,
            )
        except GraphError:
            return []
        # SSpaths yields every prefix of the chain as its own path. The longest
        # one contains all the others, so the history is that path's nodes in
        # order rather than the union of every row.
        best: list[dict[str, Any]] = []
        for row in rows:
            nodes = (row.values.get("path") or {}).get("nodes", [])
            if len(nodes) > len(best):
                best = nodes
        return [
            {**(node.get("properties") or {}), "node": node.get("id")}
            for node in best
            if isinstance(node, dict)
        ]

    def blast_radius(self, claim_id: str, limit: int = 60) -> list[dict[str, Any]]:
        """Who has been reading the version that turned out to be wrong.

        Claim to the document that asserts it, then back out to everyone the
        ontology connects to that document -- named in it, speaking in it, or
        sending it. Three relationship types rather than one because they are
        three different degrees of exposure: the person who sent the mail
        stating a superseded limit is in a different position from someone
        merely mentioned in it, and collapsing them would overstate the blast.

        Anchored on the claim, so the `Document` label being half a million
        nodes costs nothing -- the traversal never scans it.
        """
        anchor = node_id(f"claim:{claim_id}")
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for rel, how in (
            ("SENT", "sent"),
            ("SPOKE_IN", "spoke in"),
            ("MENTIONED_IN", "named in"),
        ):
            try:
                rows = self.query(
                    f"MATCH (c:Claim {{id: {anchor}}})-[:EVIDENCED_BY]->(d:Document)"
                    f"<-[:{rel}]-(e:Entity) "
                    "RETURN e.eid AS eid, e.canonical_name AS name, e.id AS node, "
                    f"d.doc_id AS doc_id, d.title AS title, d.source AS source "
                    f"LIMIT {int(limit)}",
                    strong=True,
                )
            except GraphError:
                continue
            for row in rows:
                v = row.values
                eid, doc_id = str(v.get("eid") or ""), str(v.get("doc_id") or "")
                if not eid or (eid, doc_id) in seen:
                    continue
                seen.add((eid, doc_id))
                out.append(
                    {
                        "eid": eid,
                        "name": str(v.get("name") or ""),
                        "node": int(v.get("node") or 0),
                        "doc_id": doc_id,
                        "title": str(v.get("title") or ""),
                        "source": str(v.get("source") or ""),
                        "how": how,
                    }
                )
        # Strongest connection first: sending a document is evidence you meant
        # what it says, being named in one is not.
        order = {"sent": 0, "spoke in": 1, "named in": 2}
        return sorted(out, key=lambda r: (order[r["how"]], r["name"]))[:limit]

    def contradicted_by(self, claim_id: str, limit: int = 12) -> list["ClaimNode"]:
        """The claims that disagree with this one, straight off the edge."""
        try:
            rows = self.query(
                f"MATCH (a:Claim {{id: {node_id('claim:' + claim_id)}}})"
                "-[r:CONTRADICTS]->(c:Claim) RETURN "
                + _CLAIM_FIELDS
                + f" ORDER BY c.trust DESC LIMIT {int(limit)}",
                strong=True,
            )
        except GraphError:
            return []
        return [_claim(row.values) for row in rows]

    def claims_about(self, eid: str, limit: int = 40) -> list["ClaimNode"]:
        """Every claim wired to one person or thing through `ABOUT`.

        The join between the two halves of the graph: the subject of a claim is
        a written form, and the identity graph already knows who written forms
        denote, so a claim about "Jordan" and a claim about "J. Reyes" are
        reachable from the same anchor.
        """
        try:
            rows = self.query(
                f"MATCH (c:Claim)-[:ABOUT]->(e:Entity {{id: {node_id('entity:' + eid)}}}) "
                "RETURN " + _CLAIM_FIELDS + f" ORDER BY c.trust DESC LIMIT {int(limit)}",
                strong=True,
            )
        except GraphError:
            return []
        return [_claim(row.values) for row in rows]

    def claim_stats(self) -> dict[str, int]:
        """Sizes of the contradiction graph, for the header and for the loader.

        `count(*)` is the one aggregate the parser accepts, and it is only safe
        on labels this small.
        """
        out: dict[str, int] = {}
        for key, cypher in (
            ("claims", "MATCH (n:Claim) RETURN count(*)"),
            ("disagreements", "MATCH (n:Disagreement) RETURN count(*)"),
            # Both endpoints must be bound. An anonymous `(:Claim)` is
            # rejected with "node label patterns are not supported yet", which
            # is a parser limitation rather than anything about the data.
            ("contradicts", "MATCH (a:Claim)-[:CONTRADICTS]->(b:Claim) RETURN count(*)"),
            ("supersedes", "MATCH (a:Claim)-[:SUPERSEDES]->(b:Claim) RETURN count(*)"),
        ):
            try:
                rows = self.query(cypher, strong=True, timeout=30)
                out[key] = int(next(iter(rows[0].values.values()))) if rows else 0
            except GraphError:
                out[key] = 0
        return out
