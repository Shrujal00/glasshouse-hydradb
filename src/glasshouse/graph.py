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
        sets = ", ".join(f"r.{p} = row.{p}" for p in properties)
        self.query(
            f"UNWIND $rows AS row "
            f"MATCH (s:{src_label} {{id: row.src}}), (d:{dst_label or src_label} {{id: row.dst}}) "
            f"MERGE (s)-[r:{rel} {{id: row.id}}]->(d) "
            f"SET {sets}",
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
