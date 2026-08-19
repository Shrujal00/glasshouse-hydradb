"""The graph's third entrance: the place a document lives, not the people in it.

`metadata` questions score 0% and 77 of the 100 answers are sitting in the
expected document's text, so the failure is reaching that document at all.
Those questions name a folder, a channel, a speaker or a sender -- structure
the normalized records carry and the graph did not. These tests pin the shapes
that structure is written in, and pin that retrieval through it fails soft
while the load is still an open decision.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

from glasshouse.graph import GraphEngine, GraphError, Row, node_id
from glasshouse.priors import Priors

SPEC = importlib.util.spec_from_file_location(
    "load_facet_graph", Path(__file__).parents[1] / "scripts" / "load_facet_graph.py"
)
loader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(loader)


class FakeEngine:
    """Captures writes the way the document loader's fake does, per relationship."""

    def __init__(self) -> None:
        self.containers: dict[int, dict] = {}
        self.edges: dict[str, dict[int, dict]] = {}
        self.batches: list[tuple[str, int]] = []

    def wait_until_ready(self, timeout=90) -> bool:
        return True

    def upsert_nodes(self, label, rows, properties) -> None:
        assert label == "Container"
        assert list(properties) == ["key", "source", "kind", "name", "documents"]
        self.batches.append((label, len(rows)))
        self.containers.update({row["id"]: dict(row) for row in rows})

    def merge_edges(self, rel, rows, properties, src_label, dst_label=None) -> None:
        assert (rel, src_label, dst_label) in {
            ("IN_CONTAINER", "Document", "Container"),
            ("SPOKE_IN", "Entity", "Document"),
            ("SENT", "Entity", "Document"),
        }
        self.batches.append((rel, len(rows)))
        for row in rows:
            key = node_id(f"{row['src']}-{rel}->{row['dst']}")
            self.edges.setdefault(rel, {})[key] = dict(row)


def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    lookup = tmp_path / "ontology.sqlite3"
    conn = sqlite3.connect(lookup)
    conn.execute(
        "CREATE TABLE alias (surface, kind, eid, node_id, canonical_name, confidence, alias_count)"
    )
    conn.executemany(
        "INSERT INTO alias VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("maya.chen@redwood.ai", "email", "maya", 11, "Maya Chen", 1, 3),
            ("maya chen", "name", "maya", 11, "Maya Chen", 1, 3),
            ("mchen", "handle", "maya", 11, "Maya Chen", 1, 3),
            ("jordan", "name", "jordan-one", 12, "Jordan Price", 1, 1),
            ("jordan", "name", "jordan-two", 13, "Jordan Reyes", 1, 1),
        ],
    )
    conn.commit()
    return normalized, lookup, tmp_path / "checkpoint.json"


def write_shard(normalized: Path, name: str, records: list[dict]) -> None:
    (normalized / f"{name}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def run(normalized, lookup, checkpoint, engine, sources, **kwargs):
    return loader.run(
        sources,
        normalized=normalized,
        lookup_path=lookup,
        checkpoint_path=checkpoint,
        engine=engine,
        priors=Priors(),
        **kwargs,
    )


# --- the loader --------------------------------------------------------------


def test_containers_carry_their_size_and_documents_link_to_every_one(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        "confluence",
        [
            {
                "doc_id": "d1",
                "source": "confluence",
                "folders": ["customer-success-and-support", "integration-guides"],
                "channels": ["Inc-TENANT123"],
            },
            {"doc_id": "d2", "source": "confluence", "folders": ["customer-success-and-support"]},
        ],
    )
    engine = FakeEngine()
    counts = run(normalized, lookup, checkpoint, engine, ["confluence"])

    assert counts["containers"] == 3
    assert counts["memberships"] == 4
    by_key = {row["key"]: row for row in engine.containers.values()}
    assert set(by_key) == {
        "confluence:folder:customer-success-and-support",
        "confluence:folder:integration-guides",
        # The key is case-folded so a question naming `#inc-tenant123` in lower
        # case still hashes to the node the loader wrote.
        "confluence:channel:inc-tenant123",
    }
    support = by_key["confluence:folder:customer-success-and-support"]
    assert support["documents"] == 2
    assert support["kind"] == "folder"
    assert support["name"] == "customer-success-and-support"
    assert support["id"] == node_id("container:confluence:folder:customer-success-and-support")

    memberships = list(engine.edges["IN_CONTAINER"].values())
    assert len(memberships) == 4
    assert {row["src"] for row in memberships} == {node_id("doc:d1"), node_id("doc:d2")}
    assert {row["kind"] for row in memberships} == {"folder", "channel"}
    assert all(row["dst"] in engine.containers for row in memberships)


def test_speakers_and_senders_become_separate_edges_from_mentions(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        "gmail",
        [
            {
                "doc_id": "mail",
                "source": "gmail",
                "folders": ["maya_chen"],
                "headers": {
                    "from": "Maya Chen <maya.chen@redwood.ai>",
                    "to": "Jordan <jordan@redwood.ai>",
                },
                # Mentioned, not speaking: this is the MENTIONED_IN loader's
                # business and must not become a role edge here.
                "mentions": ["mchen"],
                "emails": ["maya.chen@redwood.ai", "jordan@redwood.ai"],
                "named_emails": [{"name": "Maya Chen", "email": "maya.chen@redwood.ai"}],
            },
            {"doc_id": "call", "source": "gmail", "speakers": ["Maya Chen", "Jordan"]},
        ],
    )
    engine = FakeEngine()
    counts = run(normalized, lookup, checkpoint, engine, ["gmail"])

    sent = list(engine.edges["SENT"].values())
    assert len(sent) == 1
    assert sent[0]["src"] == 11
    assert sent[0]["dst"] == node_id("doc:mail")
    # Address and display name are the same person, so one edge with both kinds.
    assert sent[0]["kinds"] == "email,name"
    assert sent[0]["mention_count"] == 2
    # `to:` is not a sender, and a mention is not a speaker.
    assert counts["sent"] == 1

    spoke = list(engine.edges["SPOKE_IN"].values())
    assert len(spoke) == 1
    assert spoke[0]["src"] == 11
    assert spoke[0]["dst"] == node_id("doc:call")
    # "Jordan" is two people in the ontology; an ambiguous speaker is dropped
    # rather than guessed, because the trace would read a wrong edge as
    # established authorship.
    assert counts["spoke_in_ambiguous"] == 1
    assert counts["spoke_in"] == 1


def test_only_normalized_identity_fields_are_linked(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        "slack",
        [
            {
                "doc_id": "body-only",
                "source": "slack",
                "title": "Maya Chen owns this",
                "body": "Maya Chen <maya.chen@redwood.ai> said the limit is 40.",
                "channels": ["incidents"],
            }
        ],
    )
    engine = FakeEngine()
    counts = run(normalized, lookup, checkpoint, engine, ["slack"])
    assert counts["spoke_in"] == 0
    assert counts["sent"] == 0
    assert engine.edges.get("SPOKE_IN") is None
    assert engine.edges.get("SENT") is None
    assert counts["memberships"] == 1


def test_writes_batch_at_a_thousand(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        "slack",
        [
            {"doc_id": f"d{i}", "source": "slack", "channels": [f"c{i}"], "speakers": ["Maya Chen"]}
            for i in range(1500)
        ],
    )
    engine = FakeEngine()
    counts = run(normalized, lookup, checkpoint, engine, ["slack"])
    assert counts["memberships"] == 1500
    assert counts["spoke_in"] == 1500
    assert max(size for _label, size in engine.batches) <= loader.BATCH
    # 1500 containers is two node batches, and no edge batch exceeds the cap.
    assert [size for label, size in engine.batches if label == "Container"] == [1000, 500]


def test_replay_is_idempotent_and_the_checkpoint_resumes(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        "slack",
        [
            {"doc_id": f"d{i}", "source": "slack", "channels": ["incidents"], "speakers": ["Maya Chen"]}
            for i in range(3)
        ],
    )
    engine = FakeEngine()
    args = (normalized, lookup, checkpoint, engine, ["slack"])

    run(*args, limit=2)
    first = (dict(engine.containers), {rel: dict(rows) for rel, rows in engine.edges.items()})
    run(*args, limit=2)
    assert (engine.containers, engine.edges) == first
    # A bounded slice is a reproducible smoke test and leaves no progress state.
    assert not checkpoint.exists()

    run(*args)
    assert json.loads(checkpoint.read_text()) == {
        "sources": ["slack"],
        "source": "slack",
        "line": 3,
    }
    assert len(engine.edges["IN_CONTAINER"]) == 3
    assert len(engine.edges["SPOKE_IN"]) == 3
    assert engine.containers[node_id("container:slack:channel:incidents")]["documents"] == 3

    engine.batches.clear()
    run(*args)
    # Container sizes are recomputed from the shard rather than accumulated, so
    # a resumed run rewrites the same number instead of tripling it.
    assert engine.containers[node_id("container:slack:channel:incidents")]["documents"] == 3
    assert [label for label, _ in engine.batches] == ["Container"]
    assert len(engine.edges["IN_CONTAINER"]) == 3


def test_resume_skips_completed_sources_only_for_the_same_source_order(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(normalized, "slack", [{"doc_id": "d1", "source": "slack", "channels": ["incidents"]}])
    write_shard(normalized, "before", [{"doc_id": "b1", "source": "before", "folders": ["design"]}])
    checkpoint.write_text(
        json.dumps({"sources": ["before", "slack"], "source": "slack", "line": 1}) + "\n",
        encoding="utf-8",
    )
    engine = FakeEngine()
    run(normalized, lookup, checkpoint, engine, ["before", "slack"])
    # `before` is already done and `slack` is at its end, so only the container
    # upsert for `slack` runs and no edge is rewritten.
    assert engine.edges == {}
    assert {row["key"] for row in engine.containers.values()} == {"slack:channel:incidents"}

    engine = FakeEngine()
    run(normalized, lookup, checkpoint, engine, ["slack", "before"])
    assert {row["src"] for row in engine.edges["IN_CONTAINER"].values()} == {node_id("doc:b1")}


# --- the traversal ----------------------------------------------------------


def _membership(container_node: int, document_node: int, doc_id: str) -> Row:
    """One SSpaths row as the engine returns it, read from the container inwards."""
    return Row(
        {
            "path": {
                "nodes": [
                    {
                        "id": container_node,
                        "labels": ["Container"],
                        "properties": {"key": "slack:channel:incidents", "documents": 2},
                    },
                    {
                        "id": document_node,
                        "labels": ["Document"],
                        "properties": {"doc_id": doc_id},
                    },
                ],
                "relationships": [
                    {"edge_type": "IN_CONTAINER", "src": document_node, "dst": container_node}
                ],
            }
        }
    )


def test_container_scope_is_one_anchored_hop_per_container():
    class Recording(GraphEngine):
        def __init__(self):
            self.calls = []

        def query(self, cypher, parameters=None, **kwargs):
            self.calls.append(cypher)
            node = int(cypher.split("sourceNode: ")[1].split(",")[0])
            rows = [_membership(node, 50, "shared")]
            if node == 1:
                rows.append(_membership(node, 51, "folder-only"))
            return rows

    engine = Recording()
    found = engine.documents_in_containers([("space", 1), ("channel", 2)], limit=4)

    assert len(engine.calls) == 2
    # The same idiom the entity entrance uses: a labeled MATCH expansion scans
    # the whole edge set and blows the engine's 30s cap.
    assert all("algo.SSpaths" in call and "MATCH" not in call for call in engine.calls)
    assert all("relDirection: 'incoming'" in call for call in engine.calls)
    assert all("IN_CONTAINER" in call for call in engine.calls)

    shared = next(candidate for candidate in found if candidate.doc_id == "shared")
    assert shared.seed_eids == ("channel", "space")
    assert shared.hops == 1
    assert shared.reason == "direct Document-[:IN_CONTAINER]->Container membership"
    assert {path["seed_eid"] for path in shared.path["paths"]} == {"space", "channel"}
    edge = shared.path["paths"][0]["relationships"][0]
    assert (edge["edge_type"], edge["src"], edge["dst"]) == ("IN_CONTAINER", 50, 1)
    assert {candidate.doc_id for candidate in found} == {"shared", "folder-only"}


def test_a_huge_container_does_not_consume_the_whole_scope():
    class Lopsided(GraphEngine):
        """#incidents holds 28,999 documents; the other container holds one."""

        def __init__(self):
            pass

        def query(self, cypher, parameters=None, **kwargs):
            node = int(cypher.split("sourceNode: ")[1].split(",")[0])
            if node == 1:
                return [_membership(node, i, f"incident-{i}") for i in range(50)]
            return [_membership(node, 900, "the-one")]

    found = Lopsided().documents_in_containers([("incidents", 1), ("small", 2)], limit=4)
    assert len(found) == 4
    assert "the-one" in {candidate.doc_id for candidate in found}


def test_missing_containers_return_empty_rather_than_raising():
    class Unloaded(GraphEngine):
        """The container half of the graph is not loaded yet."""

        def __init__(self):
            pass

        def query(self, cypher, parameters=None, **kwargs):
            return []

    class Rejecting(GraphEngine):
        def __init__(self):
            pass

        def query(self, cypher, parameters=None, **kwargs):
            raise GraphError("429 resource_exhausted")

    # An absent anchor yields no paths, and a rejected query is skipped, because
    # the local FacetStore can still serve this scope and the answer must not
    # die waiting for a load the lead has not authorised.
    assert Unloaded().documents_in_containers([("space", 1)], limit=4) == []
    assert Rejecting().documents_in_containers([("space", 1)], limit=4) == []
    assert Unloaded().documents_in_containers([], limit=4) == []
    assert Unloaded().documents_in_containers([("space", 1)], limit=0) == []
