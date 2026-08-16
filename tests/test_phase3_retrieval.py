from __future__ import annotations

import sqlite3

from glasshouse import answer
from glasshouse.answer import Written
from glasshouse.ask import Asker, Person
from glasshouse.graph import GraphCandidate, GraphEngine, Row
from glasshouse.priors import Priors
from glasshouse.recall import Candidate, LocalRecall


class GraphScope:
    def documents_for_entities(self, seeds, limit):
        assert seeds == [("sam", 101)]
        assert limit > 1
        return [
            GraphCandidate(
                "answer",
                ("sam",),
                {"nodes": [{"id": 101}, {"id": 201}]},
                1,
                "direct Entity-[:MENTIONED_IN]->Document reachability",
            ),
            GraphCandidate(
                "unrelated",
                ("sam",),
                {"nodes": [{"id": 101}, {"id": 202}]},
                1,
                "direct Entity-[:MENTIONED_IN]->Document reachability",
            ),
        ]


def make_phase3_asker(tmp_path):
    recall = LocalRecall(tmp_path / "recall.sqlite3")
    recall.create()
    recall.add(
        [
            (
                "decoy",
                "slack",
                "Retention policy decision",
                "S. Ratnaparkhi discussed a retention policy decision.",
                "",
                "",
                "",
            ),
            (
                "answer",
                "slack",
                "Data retention",
                "The retention period is thirty days.",
                "",
                "",
                "",
            ),
            (
                "unrelated",
                "slack",
                "Sam notes",
                "Sam attended an unrelated meeting.",
                "",
                "",
                "",
            ),
        ]
    )
    recall.optimize()

    lookup = sqlite3.connect(tmp_path / "ontology.sqlite3")
    lookup.row_factory = sqlite3.Row
    lookup.execute(
        "CREATE TABLE alias (surface, kind, eid, node_id, canonical_name, confidence, alias_count)"
    )
    lookup.executemany(
        "INSERT INTO alias VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("sam", "handle", "sam", 101, "S. Ratnaparkhi", 1.0, 4),
            ("sam ratnaparkhi", "name", "sam", 101, "S. Ratnaparkhi", 1.0, 4),
            ("s. ratnaparkhi", "name", "sam", 101, "S. Ratnaparkhi", 1.0, 4),
            ("sam@redwood.ai", "email", "sam", 101, "S. Ratnaparkhi", 1.0, 4),
            ("sam lee", "name", "sam-lee", 102, "Sam Lee", 1.0, 1),
            ("sam patel", "name", "sam-patel", 103, "Sam Patel", 1.0, 1),
        ],
    )
    asker = Asker.__new__(Asker)
    asker.recall = recall
    asker.engine = GraphScope()
    asker._lookup = lookup
    asker.priors = Priors()
    return asker


def test_graph_scoped_retrieval_adds_alias_document_and_preserves_provenance(tmp_path):
    asker = make_phase3_asker(tmp_path)

    result = asker.retrieve("What did Sam decide about the retention policy?", limit=2)

    assert [person.eid for person in result.named_entities] == ["sam"]
    assert "answer" not in {doc.doc_id for doc in result.plain_docs}
    assert "answer" not in {doc.doc_id for doc in result.identity_docs}
    assert result.graph_docs[0].doc_id == "answer"
    assert "answer" in {doc.doc_id for doc in result.final_docs}
    assert result.graph_candidates[0].seed_eids == ("sam",)
    assert result.graph_candidates[0].path["nodes"][-1]["id"] == 201
    assert "sam" not in asker.recall.topic_terms(
        "What did @sam decide about retention?", mute=["@sam"]
    )


def test_query_without_a_named_entity_uses_plain_retrieval_only(tmp_path):
    asker = make_phase3_asker(tmp_path)
    result = asker.retrieve("retention policy", limit=2)
    assert result.named_entities == []
    assert result.graph_candidates == []
    assert result.graph_docs == []
    assert result.identity_docs == result.plain_docs


def test_stream_and_non_streaming_emit_graph_provenance_and_use_final_docs(
    tmp_path, monkeypatch
):
    question = "What did Sam decide about the retention policy?"
    asker = make_phase3_asker(tmp_path)
    monkeypatch.setattr(
        answer,
        "write_streaming",
        lambda *args, **kwargs: iter([{"done": Written("Thirty days [2]", False, [2])}]),
    )
    streamed = list(asker.stream(question, limit=2))
    scope = next(event for event in streamed if event["type"] == "graph_scope")
    added = [event for event in streamed if event["type"] == "graph_document"]
    ablation = next(event for event in streamed if event["type"] == "graph_ablation")
    assert scope == {
        "type": "graph_scope",
        "entities": ["sam"],
        "candidates": 2,
        "available": True,
    }
    assert added[0]["doc_id"] == "answer"
    assert added[0]["path"]["nodes"][-1]["id"] == 201
    assert ablation["graph_only"] == 1

    named = asker.read_identities(question)
    monkeypatch.setattr(asker, "identify", lambda docs: {})
    monkeypatch.setattr(asker, "resolve", lambda surfaces: named)

    def write(question, docs, people, paths):
        answer_number = next(i for i, doc in enumerate(docs, 1) if doc.doc_id == "answer")
        return Written(f"Thirty days [{answer_number}]", False, [answer_number])

    monkeypatch.setattr(answer, "write", write)
    result = asker.ask(question, limit=2)
    assert result.cited
    assert result.documents[result.cited[0] - 1].doc_id == "answer"
    assert {event.kind for event in result.events} >= {
        "graph_scope",
        "graph_document",
        "graph_ablation",
    }

    monkeypatch.setattr(answer, "write", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model offline")))
    degraded = asker.ask(question, limit=2)
    assert degraded.abstained == "answer synthesis failed (RuntimeError)"
    assert any(event.kind == "answer_degraded" for event in degraded.events)


def test_graph_failure_is_explicitly_reported(tmp_path, monkeypatch):
    class BrokenGraph:
        def documents_for_entities(self, seeds, limit):
            raise RuntimeError("offline")

    asker = make_phase3_asker(tmp_path)
    asker.engine = BrokenGraph()
    result = asker.retrieve(
        "What did Sam decide about the retention policy?", limit=2
    )
    assert result.graph_candidates == []
    assert result.graph_error == "RuntimeError: offline"

    monkeypatch.setattr(
        answer,
        "write_streaming",
        lambda *args, **kwargs: iter([{"done": Written("Related [1]", False, [1])}]),
    )
    events = list(asker.stream("What did Sam decide about the retention policy?", limit=2))
    degraded = next(event for event in events if event["type"] == "degraded")
    assert degraded["step"] == "graph_retrieval"
    assert "RuntimeError" in degraded["detail"]
    assert next(event for event in events if event["type"] == "graph_ablation")[
        "available"
    ] is False


def test_final_retrieval_preserves_full_identity_and_graph_rankings():
    def candidate(doc_id):
        return Candidate(doc_id, "slack", doc_id, "", "", 1.0)

    identity_docs = [candidate("identity-1"), candidate("identity-2")]
    graph_docs = [candidate("graph-1"), candidate("graph-2")]

    class Recall:
        def search(self, question, limit, also=(), drop=()):
            return identity_docs if also else [candidate("plain")]

        def search_scoped(self, question, doc_ids, limit, drop=()):
            return graph_docs

    class Engine:
        def documents_for_entities(self, seeds, limit):
            return [
                GraphCandidate(doc.doc_id, ("sam",), {"paths": []}, 1, "direct")
                for doc in graph_docs
            ]

    asker = Asker.__new__(Asker)
    asker.recall = Recall()
    asker.engine = Engine()
    person = Person("sam", "Sam", 1, 1.0, 2, surfaces={"sam"})
    asker.read_identities = lambda question: [person]
    asker.surfaces_of = lambda person: ["sam", "sam@example.com"]
    result = asker.retrieve("What did Sam change?", limit=2)
    assert [doc.doc_id for doc in result.final_docs] == [
        "identity-1",
        "identity-2",
        "graph-1",
        "graph-2",
    ]
    assert [doc.doc_id for doc in asker._answer_documents(result, 2)] == [
        "identity-1",
        "graph-1",
    ]


def test_get_many_preserves_requested_order(tmp_path):
    recall = LocalRecall(tmp_path / "recall.sqlite3")
    recall.create()
    recall.add([
        ("one", "slack", "One", "", "", "", ""),
        ("two", "slack", "Two", "", "", "", ""),
    ])
    recall.optimize()
    assert [doc.doc_id for doc in recall.get_many(["two", "missing", "one"])] == [
        "two",
        "one",
    ]


def test_scoped_ranking_backfills_after_higher_coverage_duplicates(tmp_path):
    recall = LocalRecall(tmp_path / "recall.sqlite3")
    recall.create()
    rows = [
        (f"both-{i}", "slack", "alpha beta", "", "", "", "")
        for i in range(10)
    ] + [
        (f"one-{i}", "slack", "alpha", "", "", "", "")
        for i in range(20)
    ]
    recall.add(rows)
    recall.optimize()
    docs = recall.search_scoped(
        "alpha beta", [row[0] for row in rows], limit=20
    )
    assert len(docs) == 20
    assert {doc.doc_id for doc in docs} >= {f"both-{i}" for i in range(10)}


def test_scoped_ranking_never_exceeds_limit(tmp_path):
    recall = LocalRecall(tmp_path / "recall.sqlite3")
    recall.create()
    rows = [
        (f"both-{i}", "slack", "alpha beta", "", "", "", "")
        for i in range(10)
    ] + [
        (f"one-{i}", "slack", "alpha", "", "", "", "")
        for i in range(30)
    ]
    recall.add(rows)
    recall.optimize()
    assert len(
        recall.search_scoped("alpha beta", [row[0] for row in rows], limit=20)
    ) == 20


def test_singleton_without_email_can_seed_graph_retrieval(tmp_path):
    lookup = sqlite3.connect(tmp_path / "ontology.sqlite3")
    lookup.row_factory = sqlite3.Row
    lookup.execute(
        "CREATE TABLE alias (surface, kind, eid, node_id, canonical_name, confidence, alias_count)"
    )
    lookup.executemany(
        "INSERT INTO alias VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("maya", "handle", "maya", 7, "Maya", 1.0, 1),
            ("maya", "name", "maya", 7, "Maya", 1.0, 1),
            ("maya chen", "name", "maya-chen", 10, "Maya Chen", 1.0, 1),
            ("maya lee", "name", "maya-lee", 11, "Maya Lee", 1.0, 1),
            ("maya patel", "name", "maya-patel", 12, "Maya Patel", 1.0, 1),
            ("capacity", "handle", "capacity", 8, "Cloudzone Partnerships", 0.9, 3),
            ("policy", "handle", "policy", 9, "Am Underwriters", 0.9, 3),
            ("change", "handle", "change", 13, "change", 1.0, 1),
            ("retention", "handle", "retention", 14, "Retention Analytics", 0.9, 2),
            ("retention analytics", "name", "retention", 14, "Retention Analytics", 0.9, 2),
        ],
    )
    asker = Asker.__new__(Asker)
    asker._lookup = lookup
    asker.priors = Priors()
    assert [
        (person.eid, person.node)
        for person in asker.read_identities("What did Maya change?")
    ] == [("maya", 7)]
    assert [
        person.eid for person in asker.read_identities("Maya changed what?")
    ] == ["maya"]
    assert [
        person.eid for person in asker.read_identities("What did @maya change?")
    ] == ["maya"]
    assert asker.read_identities("What changed in Capacity?") == []
    assert asker.read_identities("Policy updates") == []
    assert asker.read_identities("What changed in Change?") == []
    assert asker.read_identities("What changed in Retention?") == []


def test_direct_graph_scope_uses_supported_anchored_queries_and_merges_provenance():
    class RecordingEngine(GraphEngine):
        def __init__(self):
            self.calls = []

        def query(self, cypher, parameters=None, **kwargs):
            self.calls.append((cypher, parameters, kwargs))
            seed = parameters["id"]
            rows = [Row({"seed_id": seed, "document_node": 50, "doc_id": "shared"})]
            if seed == 1:
                rows.append(Row({"seed_id": seed, "document_node": 51, "doc_id": "first"}))
            return rows

    engine = RecordingEngine()
    candidates = engine.documents_for_entities([("one", 1), ("two", 2)], limit=4)
    assert len(engine.calls) == 2
    assert all("UNWIND" not in query for query, _, _ in engine.calls)
    assert [parameters for _, parameters, _ in engine.calls] == [{"id": 1}, {"id": 2}]
    shared = next(candidate for candidate in candidates if candidate.doc_id == "shared")
    assert shared.seed_eids == ("one", "two")
    assert shared.hops == 1
    assert {path["seed_eid"] for path in shared.path["paths"]} == {"one", "two"}
    assert all(
        path["relationships"][0]["edge_type"] == "MENTIONED_IN"
        for path in shared.path["paths"]
    )


def test_multi_seed_scope_refills_after_overlapping_neighbors():
    class OverlapEngine(GraphEngine):
        def __init__(self):
            pass

        def query(self, cypher, parameters=None, **kwargs):
            seed = parameters["id"]
            return [
                Row({"seed_id": seed, "document_node": i, "doc_id": f"shared-{i}"})
                for i in range(2)
            ] + [
                Row(
                    {
                        "seed_id": seed,
                        "document_node": seed * 10 + i,
                        "doc_id": f"seed-{seed}-{i}",
                    }
                )
                for i in range(2)
            ]

    candidates = OverlapEngine().documents_for_entities(
        [("one", 1), ("two", 2)], limit=4
    )
    assert len(candidates) == 4
    assert {candidate.doc_id for candidate in candidates} >= {
        "seed-1-0",
        "seed-2-0",
    }


def test_connect_reads_verified_paths_without_query_time_writes():
    class PathEngine:
        def paths(self, source, target, rel_types, max_len, path_count):
            assert (source, target) == (1, 2)
            assert rel_types == ["MENTIONED_IN"]
            return [
                {
                    "path": {
                        "nodes": [
                            {"properties": {"canonical_name": "Maya"}},
                            {"properties": {"title": "Retention decision"}},
                            {"properties": {"canonical_name": "Sam"}},
                        ]
                    }
                }
            ]

    asker = Asker.__new__(Asker)
    asker.engine = PathEngine()
    people = [
        Person("maya", "Maya", 1, 1.0, 1),
        Person("sam", "Sam", 2, 1.0, 1),
    ]
    paths = asker.connect("q", people, [])
    assert paths == [
        {
            "a": "Maya",
            "b": "Sam",
            "via": ["Retention decision"],
            "summary": "Maya - Retention decision - Sam",
        }
    ]
