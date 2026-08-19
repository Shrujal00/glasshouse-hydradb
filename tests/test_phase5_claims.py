"""Claim extraction and arbitration — the path that has to choose between two
values the model can already see.

Retrieval is not the problem on `conflicting_info`: plain FTS puts the expected
document in the synthesis context on 10 of 10 of those questions. The model
still scores badly because it is handed two competing numbers and no way to
prefer one. These tests pin the two halves of the fix: extraction that invents
nothing, and arbitration that decides deterministically and says which signal
decided it.

Nothing here touches the network. Extraction is driven through a stub client
that returns whatever text a test wants the model to have said.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from glasshouse import claims as claims_module
from glasshouse.claims import Claim, extract
from glasshouse.recall import Candidate
from glasshouse.trust import arbitrate, score


def document(body: str, doc_id: str = "d1", source: str = "confluence") -> Candidate:
    return Candidate(
        doc_id=doc_id,
        source=source,
        title="Audit log shipper",
        body=body,
        date="2026-03-10",
        score=1.0,
    )


class StubClient:
    """A model that says exactly what the test tells it to, and counts calls."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs["messages"])
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return {"message": {"content": reply}}


def stub(monkeypatch, *replies: str) -> StubClient:
    client = StubClient(*replies)
    monkeypatch.setattr(claims_module, "_client", lambda: client)
    return client


def payload(**fields) -> str:
    base = {
        "document": 1,
        "subject": "audit-log shipper",
        "predicate": "owner",
        "object_value": "Priya Nair",
        "confidence": 0.9,
    }
    base.update(fields)
    return json.dumps({"claims": [base]})


def claim(
    subject: str = "PM-772222",
    predicate: str = "status",
    value: str = "in review",
    doc_id: str = "d1",
    source: str = "confluence",
    asserted_at: str = "2026-03-10",
    confidence: float = 0.9,
) -> Claim:
    return Claim(
        claim_id=f"{doc_id}:{predicate}:{value}",
        subject=subject,
        predicate=predicate,
        object_value=value,
        doc_id=doc_id,
        source=source,
        asserted_at=asserted_at,
        extractor_confidence=confidence,
    )


# --- extraction -------------------------------------------------------------


def test_extraction_returns_a_claim_grounded_in_the_document(monkeypatch, tmp_path):
    stub(monkeypatch, payload())
    found = extract(
        [document("The audit-log shipper is owned by Priya Nair.")],
        "who owns the audit-log shipper?",
        cache_path=tmp_path / "claims.sqlite3",
    )
    assert len(found) == 1
    assert (found[0].subject, found[0].predicate, found[0].object_value) == (
        "audit-log shipper",
        "owner",
        "Priya Nair",
    )
    # Provenance comes from the retrieved document, never from the model: the
    # one field a model will happily fabricate is where it read something.
    assert found[0].doc_id == "d1"
    assert found[0].source == "confluence"
    assert found[0].asserted_at == "2026-03-10"


def test_prose_instead_of_json_produces_no_claim_and_no_exception(monkeypatch, tmp_path):
    stub(monkeypatch, "I could not find any owners in these documents, sorry!")
    assert extract(
        [document("The audit-log shipper is owned by Priya Nair.")],
        "who owns it?",
        cache_path=tmp_path / "claims.sqlite3",
    ) == []


def test_malformed_output_gets_exactly_one_repair_attempt(monkeypatch, tmp_path):
    client = stub(monkeypatch, "{\"claims\": [", "{\"claims\": [")
    assert extract(
        [document("The audit-log shipper is owned by Priya Nair.")],
        "who owns it?",
        cache_path=tmp_path / "claims.sqlite3",
    ) == []
    assert len(client.calls) == 2


def test_a_repair_that_returns_json_is_accepted(monkeypatch, tmp_path):
    client = stub(monkeypatch, "here you go: {\"claims\":", payload())
    found = extract(
        [document("The audit-log shipper is owned by Priya Nair.")],
        "who owns it?",
        cache_path=tmp_path / "claims.sqlite3",
    )
    assert len(client.calls) == 2
    assert [c.object_value for c in found] == ["Priya Nair"]


def test_unknown_predicates_are_dropped_rather_than_normalized(monkeypatch, tmp_path):
    stub(monkeypatch, payload(predicate="escalation_path", object_value="Priya Nair"))
    assert extract(
        [document("The audit-log shipper escalates to Priya Nair.")],
        "who owns it?",
        cache_path=tmp_path / "claims.sqlite3",
    ) == []


def test_a_value_absent_from_the_evidence_is_dropped(monkeypatch, tmp_path):
    """The failure mode this whole module is defensive about: a plausible name
    that is nowhere in the text the model was shown."""
    stub(monkeypatch, payload(object_value="Jordan Reyes"))
    assert extract(
        [document("The audit-log shipper is owned by Priya Nair.")],
        "who owns it?",
        cache_path=tmp_path / "claims.sqlite3",
    ) == []


def test_a_claim_pointing_at_no_supplied_document_is_dropped(monkeypatch, tmp_path):
    stub(monkeypatch, payload(document=7))
    assert extract(
        [document("The audit-log shipper is owned by Priya Nair.")],
        "who owns it?",
        cache_path=tmp_path / "claims.sqlite3",
    ) == []


def test_a_model_failure_degrades_to_no_claims(monkeypatch, tmp_path):
    def explode():
        raise RuntimeError("model offline")

    monkeypatch.setattr(claims_module, "_client", explode)
    assert extract(
        [document("owned by Priya Nair")], "who owns it?",
        cache_path=tmp_path / "claims.sqlite3",
    ) == []


def test_the_same_document_is_not_extracted_twice(monkeypatch, tmp_path):
    cache = tmp_path / "claims.sqlite3"
    client = stub(monkeypatch, payload())
    docs = [document("The audit-log shipper is owned by Priya Nair.")]
    first = extract(docs, "who owns the audit-log shipper?", cache_path=cache)
    second = extract(docs, "who owns the audit-log shipper?", cache_path=cache)
    assert len(client.calls) == 1
    assert [c.object_value for c in second] == [c.object_value for c in first]


def test_a_failed_extraction_is_not_cached_as_an_answer(monkeypatch, tmp_path):
    cache = tmp_path / "claims.sqlite3"
    docs = [document("The audit-log shipper is owned by Priya Nair.")]
    stub(monkeypatch, "not json at all")
    assert extract(docs, "who owns it?", cache_path=cache) == []
    client = stub(monkeypatch, payload())
    assert len(extract(docs, "who owns it?", cache_path=cache)) == 1
    assert len(client.calls) == 1


# --- corroboration versus conflict -----------------------------------------


def test_the_same_value_twice_is_corroboration_not_conflict():
    result = arbitrate([
        claim(value="in review", doc_id="d1", source="confluence"),
        claim(value="In Review.", doc_id="d2", source="jira"),
    ])
    assert result.conflicts == ()
    assert {c.status for c in result.claims} == {"accepted"}
    # Corroboration is a trust signal, so agreement has to raise trust above
    # what either document earns alone.
    assert all(c.trust > score(c) - 1e-9 for c in result.claims)
    assert "2 documents" in result.render() or "corroborat" in result.render().lower()


def test_different_values_for_one_subject_and_predicate_conflict():
    result = arbitrate([
        claim(value="in review", doc_id="d1", source="confluence"),
        claim(value="blocked", doc_id="d2", source="slack"),
    ])
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    # The grouping key is punctuation-free, so `PM-772222` and `PM 772222`
    # cannot end up in two groups that never see each other.
    assert (conflict.subject, conflict.predicate) == ("pm 772222", "status")
    assert conflict.winner.object_value == "in review"
    assert [loser.object_value for loser in conflict.losers] == ["blocked"]


def test_a_different_subject_is_not_a_conflict():
    result = arbitrate([
        claim(subject="PM-772222", value="in review"),
        claim(subject="PM-993311", value="blocked", doc_id="d2"),
    ])
    assert result.conflicts == ()


# --- what decides ----------------------------------------------------------


def test_a_newer_claim_supersedes_an_older_one_on_a_volatile_predicate():
    result = arbitrate([
        claim(value="blocked", doc_id="d1", source="confluence", asserted_at="2026-01-05"),
        claim(value="shipped", doc_id="d2", source="slack", asserted_at="2026-04-02"),
    ])
    conflict = result.conflicts[0]
    assert conflict.winner.object_value == "shipped"
    assert "2026-04-02" in conflict.rationale
    losers = {c.claim_id: c for c in result.claims if c.status != "accepted"}
    assert losers[conflict.losers[0].claim_id].status == "superseded"


def test_a_stable_predicate_prefers_the_authoritative_corroborated_claim():
    result = arbitrate([
        claim(predicate="reports_to", value="Dana Whitfield", doc_id="d1",
              source="confluence", asserted_at="2026-01-05"),
        claim(predicate="reports_to", value="Dana Whitfield", doc_id="d2",
              source="github", asserted_at="2026-01-06"),
        claim(predicate="reports_to", value="Marcus Feld", doc_id="d3",
              source="slack", asserted_at="2026-04-02"),
    ])
    conflict = result.conflicts[0]
    assert conflict.winner.object_value == "Dana Whitfield"
    # Recency pointed the other way and must not be claimed as the reason.
    assert "2026-04-02" not in conflict.rationale
    assert "later" not in conflict.rationale
    assert "corroborat" in conflict.rationale


def test_the_rationale_names_only_signals_that_fired():
    """Two claims from the same source on the same date, differing only in how
    confident extraction was: authority and recency did not separate them, so
    the sentence must not mention either."""
    result = arbitrate([
        claim(value="in review", doc_id="d1", source="slack",
              asserted_at="2026-04-02", confidence=0.9),
        claim(value="blocked", doc_id="d2", source="slack",
              asserted_at="2026-04-02", confidence=0.4),
    ])
    rationale = result.conflicts[0].rationale.lower()
    assert "confidence" in rationale
    assert "authority" not in rationale
    assert "later" not in rationale
    assert "corroborat" not in rationale


def test_a_losing_claim_stays_queryable_with_its_rationale():
    result = arbitrate([
        claim(value="in review", doc_id="d1", source="confluence"),
        claim(value="blocked", doc_id="d2", source="slack"),
    ])
    losing = [c for c in result.claims if c.object_value == "blocked"]
    assert len(losing) == 1
    assert losing[0].status in {"superseded", "disputed"}
    assert losing[0].rationale
    assert losing[0].trust > 0


# --- abstention ------------------------------------------------------------


def test_two_weak_claims_abstain_rather_than_crowning_a_winner():
    result = arbitrate([
        claim(value="probably 20%", doc_id="d1", source="slack",
              predicate="limit", asserted_at="2026-04-02", confidence=0.35),
        claim(value="maybe 30%", doc_id="d2", source="slack",
              predicate="limit", asserted_at="2026-04-02", confidence=0.3),
    ])
    conflict = result.conflicts[0]
    assert not conflict.decided
    assert result.abstains
    assert {c.status for c in result.claims} == {"disputed"}
    assert "NO ACCEPTED VALUE" in result.render()


def test_a_near_tie_abstains_even_when_both_claims_are_strong():
    result = arbitrate([
        claim(value="in review", doc_id="d1", source="confluence", asserted_at="2026-04-02"),
        claim(value="blocked", doc_id="d2", source="github", asserted_at="2026-04-02"),
    ])
    assert not result.conflicts[0].decided
    assert "too close" in result.conflicts[0].rationale


# --- what synthesis is handed ----------------------------------------------


def test_render_shows_both_values_with_source_and_date_and_the_winner():
    result = arbitrate([
        claim(value="blocked", doc_id="d1", source="confluence", asserted_at="2026-01-05"),
        claim(value="shipped", doc_id="d2", source="slack", asserted_at="2026-04-02"),
    ])
    rendered = result.render()
    assert "blocked" in rendered and "shipped" in rendered
    assert "confluence" in rendered and "slack" in rendered
    assert "2026-01-05" in rendered and "2026-04-02" in rendered
    assert "ACCEPTED" in rendered
    assert result.conflicts[0].rationale in rendered


def test_render_is_empty_when_there_is_nothing_to_arbitrate():
    assert arbitrate([]).render() == ""
