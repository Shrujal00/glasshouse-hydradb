"""The answer only ever saw the first 2600 characters of each document.

The document that answers the burst-credit question says "reserve 30%
(previous internal suggestion was 20%)" at character 4169, so the model never
saw either side of the contradiction -- only an unrelated "~30% of requests
returned 429" near the top, which invites exactly the wrong answer. Selecting
passages around the question's own terms fits the relevant part of a long
document into the same budget.
"""

from __future__ import annotations

from glasshouse.answer import build_prompt, select_passages
from glasshouse.recall import Candidate


def document(body: str, doc_id: str = "d1") -> Candidate:
    return Candidate(doc_id=doc_id, source="google_drive", title="Pool notes",
                     body=body, date="2026-03-10", score=1.0)


def test_select_passages_reaches_a_fact_past_the_head_of_the_document():
    body = (
        "Observed 30% of short chat requests returned 429 during the flush window. "
        + "filler about autoscaler metrics and signed URL callbacks. " * 90
        + "Updated reservation target: reserve 30% (previous internal suggestion "
        "was 20%) of interactive burst credits exclusively for priority=high routes."
        + " trailing unrelated closure criteria. " * 40
    )
    assert len(body) > 4000
    picked = select_passages(body, "burst credits reserved for priority=high routes", budget=1200)
    assert "previous internal suggestion" in picked
    assert len(picked) <= 1400


def test_select_passages_keeps_the_head_when_nothing_matches():
    body = "Opening statement that matters. " + ("filler. " * 400)
    picked = select_passages(body, "entirely unrelated vocabulary", budget=300)
    assert picked.startswith("Opening statement")


def test_select_passages_leaves_short_documents_whole():
    body = "A short note about burst credits."
    assert select_passages(body, "burst credits", budget=900) == body


def test_build_prompt_shows_the_matching_passage_of_a_long_document():
    body = ("preamble. " * 300) + (
        "reserve 30% of burst credits (previous internal suggestion was 20%)."
    )
    prompt = build_prompt(
        "what percent of burst credits are reserved?", [document(body)], []
    )
    assert "previous internal suggestion" in prompt
