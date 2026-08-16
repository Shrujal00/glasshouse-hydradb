from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

from glasshouse.graph import node_id
from glasshouse.priors import Priors


SPEC = importlib.util.spec_from_file_location(
    "load_document_graph", Path(__file__).parents[1] / "scripts" / "load_document_graph.py"
)
loader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(loader)


class FakeEngine:
    def __init__(self) -> None:
        self.nodes: dict[int, dict] = {}
        self.edges: dict[int, dict] = {}
        self.writes = 0

    def wait_until_ready(self, timeout=90) -> bool:
        return True

    def upsert_nodes(self, label, rows, properties) -> None:
        assert label == "Document"
        self.writes += 1
        self.nodes.update({row["id"]: dict(row) for row in rows})

    def merge_edges(self, rel, rows, properties, src_label, dst_label=None) -> None:
        assert (rel, src_label, dst_label) == ("MENTIONED_IN", "Entity", "Document")
        for row in rows:
            key = node_id(f"{row['src']}-{rel}->{row['dst']}")
            self.edges[key] = dict(row)


def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    lookup = tmp_path / "ontology.sqlite3"
    conn = sqlite3.connect(lookup)
    conn.execute("CREATE TABLE alias (surface, kind, eid, node_id, canonical_name, confidence, alias_count)")
    conn.executemany(
        "INSERT INTO alias VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("maya.chen@redwood.ai", "email", "maya", 11, "Maya Chen", 1, 3),
            ("maya chen", "name", "maya", 11, "Maya Chen", 1, 3),
            ("mchen", "handle", "maya", 11, "Maya Chen", 1, 3),
            ("sam", "handle", "sam-one", 12, "Sam One", 1, 2),
            ("sam", "handle", "sam-two", 13, "Sam Two", 1, 2),
            ("security@redwood.ai", "email", "security", 14, "Security", 1, 2),
            ("security", "handle", "security", 14, "Security", 1, 2),
        ],
    )
    conn.commit()
    return normalized, lookup, tmp_path / "checkpoint.json"


def write_shard(normalized: Path, records: list[dict]) -> None:
    (normalized / "slack.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_exact_structured_aliases_link_once_and_substrings_do_not(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        [
            {
                "doc_id": "d1", "source": "slack", "title": "T", "date": "2026-01-01",
                "emails": ["maya.chen@redwood.ai"],
                "named_emails": [{"name": "Maya Chen", "email": "maya.chen@redwood.ai"}],
                "speakers": ["Maya Chen"], "mentions": ["mchen", "sam", "security"],
                "attendees": [{"name": "Maya Chen"}],
            },
            {
                "doc_id": "d2", "source": "slack", "title": "sample", "date": "",
                "emails": [], "named_emails": [], "speakers": [], "mentions": [], "attendees": [],
            },
        ],
    )
    engine = FakeEngine()
    counts = loader.run(
        ["slack"], normalized=normalized, lookup_path=lookup, checkpoint_path=checkpoint,
        engine=engine, priors=Priors(functional=frozenset({"security"})),
    )
    assert counts["documents"] == 2
    assert counts["edges"] == 1
    assert counts["ambiguous"] == 1
    assert counts["unresolved"] == 1
    assert len(engine.edges) == 1
    edge = next(iter(engine.edges.values()))
    assert edge["src"] == 11
    assert edge["mention_count"] == 6
    assert edge["kinds"] == "email,handle,name"


def test_replay_is_idempotent_and_checkpoint_resumes(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        [
            {"doc_id": f"d{i}", "source": "slack", "title": "T", "date": "", "emails": ["maya.chen@redwood.ai"]}
            for i in range(3)
        ],
    )
    engine = FakeEngine()
    kwargs = {
        "normalized": normalized,
        "lookup_path": lookup,
        "checkpoint_path": checkpoint,
        "engine": engine,
        "priors": Priors(),
    }
    loader.run(["slack"], limit=2, **kwargs)
    first = (dict(engine.nodes), dict(engine.edges))
    loader.run(["slack"], limit=2, **kwargs)
    assert (engine.nodes, engine.edges) == first

    loader.run(["slack"], **kwargs)
    assert json.loads(checkpoint.read_text()) == {
        "sources": ["slack"],
        "source": "slack",
        "line": 3,
    }
    assert len(engine.nodes) == 3
    loader.run(["slack"], **kwargs)
    assert len(engine.nodes) == 3
    assert len(engine.edges) == 3


def test_resume_skips_completed_sources_only_for_the_same_source_order(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        [{"doc_id": "d1", "source": "slack", "title": "T", "date": "", "emails": []}],
    )
    (normalized / "before.jsonl").write_text(
        json.dumps({"doc_id": "before", "source": "before", "title": "T", "date": "", "emails": []}) + "\n",
        encoding="utf-8",
    )
    checkpoint.write_text(
        json.dumps(
            {"sources": ["before", "slack"], "source": "slack", "line": 1}
        )
        + "\n",
        encoding="utf-8",
    )
    engine = FakeEngine()
    loader.run(
        ["before", "slack"],
        normalized=normalized,
        lookup_path=lookup,
        checkpoint_path=checkpoint,
        engine=engine,
        priors=Priors(),
    )
    assert engine.writes == 0


def test_changed_source_order_does_not_skip_new_shards(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        [{"doc_id": "slack", "source": "slack", "title": "T", "date": "", "emails": []}],
    )
    (normalized / "before.jsonl").write_text(
        json.dumps(
            {"doc_id": "before", "source": "before", "title": "T", "date": "", "emails": []}
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint.write_text(
        json.dumps({"sources": ["slack"], "source": "slack", "line": 1}) + "\n",
        encoding="utf-8",
    )
    engine = FakeEngine()
    loader.run(
        ["before", "slack"],
        normalized=normalized,
        lookup_path=lookup,
        checkpoint_path=checkpoint,
        engine=engine,
        priors=Priors(),
    )
    assert {row["doc_id"] for row in engine.nodes.values()} == {"before"}


def test_checkpoint_advances_past_trailing_invalid_records(tmp_path):
    normalized, lookup, checkpoint = setup(tmp_path)
    write_shard(
        normalized,
        [
            {"doc_id": "valid", "source": "slack", "title": "T", "date": "", "emails": []},
            {"source": "slack", "title": "missing id"},
        ],
    )
    loader.run(
        ["slack"],
        normalized=normalized,
        lookup_path=lookup,
        checkpoint_path=checkpoint,
        engine=FakeEngine(),
        priors=Priors(),
    )
    assert json.loads(checkpoint.read_text()) == {
        "sources": ["slack"],
        "source": "slack",
        "line": 2,
    }
