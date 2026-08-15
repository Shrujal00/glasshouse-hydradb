from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from glasshouse import answer, server
from glasshouse.ask import Asker, Person, document_mentions
from glasshouse.priors import Priors
from glasshouse.recall import Candidate


def make_asker(tmp_path, aliases, functional=()):
    conn = sqlite3.connect(tmp_path / "ontology.sqlite3")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE alias (surface TEXT, kind TEXT, eid TEXT, node_id INTEGER, "
        "canonical_name TEXT, confidence REAL, alias_count INTEGER)"
    )
    conn.executemany("INSERT INTO alias VALUES (?, ?, ?, ?, ?, ?, ?)", aliases)
    asker = Asker.__new__(Asker)
    asker._lookup = conn
    asker.priors = Priors(functional=frozenset(functional))
    return asker


def test_read_identities_accepts_people_and_rejects_functional_role_aliases(tmp_path):
    asker = make_asker(
        tmp_path,
        [
            ("maya chen", "name", "maya", 1, "Maya Chen", 1.0, 2),
            ("maya.chen@redwood.ai", "email", "maya", 1, "Maya Chen", 1.0, 2),
            ("Security Lead", "name", "security", 2, "Security Lead", 1.0, 2),
            ("seclead@redwood.ai", "email", "security", 2, "Security Lead", 1.0, 2),
            ("Marketplace Onboarding", "name", "marketplace", 3, "Marketplace Onboarding", 1.0, 2),
            ("marketops@redwood.ai", "email", "marketplace", 3, "Marketplace Onboarding", 1.0, 2),
        ],
        functional={"seclead", "marketops"},
    )

    assert [p.eid for p in asker.read_identities("Ask Maya Chen")] == ["maya"]
    assert asker.read_identities("Ask the Security Lead") == []
    assert asker.read_identities("Ask Marketplace Onboarding") == []


def test_read_identities_rejects_spelled_out_role_alias_without_a_matching_prior(tmp_path):
    asker = make_asker(
        tmp_path,
        [
            ("security lead", "name", "security", 2, "Security Lead", 1.0, 2),
            ("security_lead@aurora-systems.com", "email", "security", 2, "Security Lead", 1.0, 2),
            ("marketplace onboarding", "name", "marketplace", 3, "Marketplace Onboarding", 1.0, 2),
            ("support@cloudmarketplace.com", "email", "marketplace", 3, "Marketplace Onboarding", 1.0, 2),
        ],
    )
    assert asker.read_identities("Ask the Security Lead") == []
    assert asker.read_identities("Ask Marketplace Onboarding") == []


def test_read_identities_rejects_ambiguous_short_name(tmp_path):
    asker = make_asker(
        tmp_path,
        [
            ("sam", "handle", "sam-one", 1, "Sam One", 1.0, 2),
            ("sam.one@redwood.ai", "email", "sam-one", 1, "Sam One", 1.0, 2),
            ("sam", "handle", "sam-two", 2, "Sam Two", 1.0, 2),
            ("sam.two@redwood.ai", "email", "sam-two", 2, "Sam Two", 1.0, 2),
        ],
    )
    assert asker.read_identities("What did sam decide?") == []


def candidate(text: str) -> Candidate:
    return Candidate(doc_id="d1", source="slack", title="", body=text, date="", score=1.0)


def test_document_linking_requires_a_parsed_exact_surface():
    person = Person("sam", "Sam", 1, 1.0, 2, surfaces={"sam"})
    assert not document_mentions(person, candidate("This is a sample document."))
    assert document_mentions(person, candidate("eng\n\nsam: confirmed"))


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["NOT_", "IN_CORPUS", " missing"], "missing"),
        (["N", "ormal answer"], "Normal answer"),
    ],
)
def test_streaming_hides_only_a_confirmed_abstention_marker(monkeypatch, chunks, expected):
    monkeypatch.setattr(
        answer,
        "_client",
        lambda: SimpleNamespace(
            chat=lambda **_: ({"message": {"content": chunk}} for chunk in chunks)
        ),
    )
    events = list(answer.write_streaming("q", [], []))
    assert "".join(event["chunk"] for event in events if "chunk" in event) == expected


def test_entity_endpoint_uses_integer_lookup_id_and_404s_unknown(monkeypatch, tmp_path):
    asker = make_asker(
        tmp_path,
        [("Maya Chen", "name", "maya", 42, "Maya Chen", 1.0, 2)],
    )
    calls = []
    asker.engine = SimpleNamespace(
        query=lambda query, parameters, strong: calls.append(parameters) or []
    )
    monkeypatch.setattr(server, "asker", lambda: asker)

    assert server.entity("maya") == {"eid": "maya", "aliases": []}
    assert calls == [{"id": 42}]
    with pytest.raises(server.HTTPException) as exc:
        server.entity("missing")
    assert exc.value.status_code == 404


def test_graph_prompt_labels_shared_documents_as_cooccurrence_only():
    prompt = answer.build_prompt("q", [], [], paths=[{"summary": "A - d - B"}])
    assert "co-occurrences" in prompt
    assert "do\n   not prove collaboration" in answer.SYSTEM.lower()
    assert "ownership" in answer.SYSTEM.lower()
