"""The contradiction graph — arbitration stored as edges rather than thrown away.

`test_phase5_claims.py` pins that the system can decide between two competing
values for one question. This pins what happens to that decision afterwards:
it becomes `(:Claim)-[:CONTRADICTS]->(:Claim)`, `(:Claim)-[:SUPERSEDES]->(:Claim)`
and a `(:Disagreement)` node that can be ranked without anyone having asked a
question first.

Three things are worth pinning here beyond "the rows come out right", because
all three are places where a graph like this quietly starts lying:

  * a disagreement must span two documents. One transcript listing three
    thresholds is one text read three times, and reporting it as the company
    contradicting itself is the failure mode the scope prompt introduced;
  * claims may only contradict each other inside one work item, or two
    unrelated tickets that both have an `owner` become a disagreement;
  * `SUPERSEDES` may only be written where recency is what actually settled
    it, because a supersession chain is a claim about time.

Nothing here touches the network or the engine. Extraction is driven through
the same stub client the claims tests use, and writes land in a fake that keeps
them the way the other loader tests do.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from glasshouse.claims import Claim, normalise_date
from glasshouse.graph import node_id
from glasshouse.trust import arbitrate, subject_key

SPEC = importlib.util.spec_from_file_location(
    "load_claims_graph", Path(__file__).parents[1] / "scripts" / "load_claims_graph.py"
)
loader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(loader)


def claim(
    value: str,
    *,
    doc_id: str = "d1",
    source: str = "confluence",
    date: str = "2026-03-10",
    subject: str = "burst credit limit",
    predicate: str = "limit",
    scope: str = "ENG-4821",
    confidence: float = 0.9,
) -> Claim:
    return Claim(
        claim_id=f"{doc_id}-{value}"[:12],
        subject=subject,
        predicate=predicate,
        object_value=value,
        doc_id=doc_id,
        source=source,
        asserted_at=date,
        extractor_confidence=confidence,
        scope=scope,
    )


def build(*claims: Claim, people: dict | None = None) -> dict[str, list[dict]]:
    arbitration = arbitrate(list(claims))
    titles = {c.doc_id: f"title of {c.doc_id}" for c in claims}
    return loader.build(arbitration, titles, people or {}, "gen1")


# --- what counts as a disagreement ------------------------------------------


def test_two_documents_disagreeing_becomes_a_disagreement_node():
    rows = build(
        claim("20%", doc_id="d1", source="confluence"),
        claim("30%", doc_id="d2", source="slack"),
    )
    assert len(rows["disagreements"]) == 1
    node = rows["disagreements"][0]
    assert node["predicate"] == "limit"
    assert node["sides"] == 2
    assert node["documents"] == 2
    assert set(node["sources"].split("|")) == {"confluence", "slack"}


def test_one_document_contradicting_itself_is_not_a_disagreement():
    """Three thresholds in one transcript is one text read three times.

    The claims are still written and still queryable -- what they do not get is
    a node on a map asserting that the organisation disagrees with itself.
    """
    rows = build(
        claim("20%", doc_id="d1", source="fireflies"),
        claim("30%", doc_id="d1", source="fireflies"),
        claim("40%", doc_id="d1", source="fireflies"),
    )
    assert rows["disagreements"] == []
    assert rows["contradicts"] == []
    assert {row["doc_id"] for row in rows["claims"]} == {"d1"}


def test_claims_in_different_work_items_never_contradict():
    """Two tickets that both have an owner are two facts, not a disagreement."""
    rows = build(
        claim("Priya", subject="owner", predicate="owner", scope="ENG-1", doc_id="d1"),
        claim("Jordan", subject="owner", predicate="owner", scope="ENG-2", doc_id="d2"),
    )
    assert rows["disagreements"] == []
    assert rows["contradicts"] == []


def test_same_value_from_two_documents_is_corroboration_not_conflict():
    rows = build(
        claim("30%", doc_id="d1", source="confluence"),
        claim("30 percent", doc_id="d2", source="slack"),
    )
    assert rows["disagreements"] == []
    assert len(rows["claims"]) == 2


# --- the edges --------------------------------------------------------------


def test_contradicts_is_written_in_both_directions():
    """A traversal that could only be walked from the winning side would make
    the losing claim a dead end, and the losing claim is the one whose blast
    radius somebody wants."""
    rows = build(
        claim("20%", doc_id="d1", source="confluence"),
        claim("30%", doc_id="d2", source="slack"),
    )
    pairs = {(row["src"], row["dst"]) for row in rows["contradicts"]}
    assert len(pairs) == 2
    (a, b), (c, d) = sorted(pairs)
    assert (a, b) == (c, d)[::-1] or (a, b) == (d, c)


def test_every_claim_is_evidenced_by_its_document():
    rows = build(
        claim("20%", doc_id="d1", source="confluence"),
        claim("30%", doc_id="d2", source="slack"),
    )
    assert len(rows["evidenced_by"]) == 2
    for row in rows["evidenced_by"]:
        assert row["dst"] in {node_id("doc:d1"), node_id("doc:d2")}


def test_about_edge_only_when_the_subject_names_somebody():
    """An unresolved subject gets no `ABOUT` edge rather than a guessed one."""
    known = build(
        claim("Priya", subject="jordan", predicate="owner", doc_id="d1"),
        claim("Sam", subject="jordan", predicate="owner", doc_id="d2", source="slack"),
        people={"jordan": ("ent_1", "Jordan Reyes")},
    )
    assert {row["dst"] for row in known["about"]} == {node_id("entity:ent_1")}

    unknown = build(
        claim("Priya", subject="jordan", predicate="owner", doc_id="d1"),
        claim("Sam", subject="jordan", predicate="owner", doc_id="d2", source="slack"),
    )
    assert unknown["about"] == []


def test_supersedes_only_where_recency_settled_it():
    """`status` is a value that is supposed to change, so the later assertion
    supersedes the earlier one and the chain is a real history."""
    rows = build(
        claim("open", predicate="status", subject="ENG-4821",
              doc_id="d1", source="slack", date="2026-01-05"),
        claim("shipped", predicate="status", subject="ENG-4821",
              doc_id="d2", source="slack", date="2026-06-20"),
    )
    assert len(rows["supersedes"]) == 1
    assert rows["supersedes"][0]["days"] > 0


def test_no_supersedes_when_arbitration_refused_to_decide():
    """An unresolved disagreement is not the history of a fact, and drawing it
    as one would be a lie about time."""
    rows = build(
        claim("maybe 20%", doc_id="d1", source="slack", date="2026-01-05",
              confidence=0.4),
        claim("roughly 30%", doc_id="d2", source="slack", date="2026-02-05",
              confidence=0.4),
    )
    assert rows["disagreements"], "the disagreement itself should still be recorded"
    assert rows["disagreements"][0]["decided"] == 0
    assert rows["supersedes"] == []


def test_reports_to_never_supersedes():
    """A newer mention of a reporting line is not evidence that it changed."""
    rows = build(
        claim("Priya", predicate="reports_to", subject="jordan",
              doc_id="d1", source="slack", date="2026-01-05"),
        claim("Sam", predicate="reports_to", subject="jordan",
              doc_id="d2", source="slack", date="2026-09-05"),
    )
    assert rows["supersedes"] == []


# --- what the map is ranked by ----------------------------------------------


def test_undecided_disagreements_outrank_settled_ones_of_the_same_size():
    """A disagreement nobody can settle between two credible sources is the one
    a person needs to see."""
    settled = build(
        claim("20%", doc_id="d1", source="slack", confidence=0.4),
        claim("30%", doc_id="d2", source="confluence", confidence=0.9),
    )["disagreements"][0]
    open_one = build(
        claim("20%", doc_id="d3", source="confluence", confidence=0.9),
        claim("30%", doc_id="d4", source="confluence", confidence=0.9),
    )["disagreements"][0]
    assert open_one["decided"] == 0 and settled["decided"] == 1
    assert open_one["weight"] > settled["weight"]


def test_disagreement_key_matches_the_group_arbitration_used():
    """A second implementation of the grouping key that drifted would silently
    split one disagreement into two."""
    rows = build(
        claim("20%", subject="The Burst-Credit Limit", doc_id="d1"),
        claim("30%", subject="burst credit limit", doc_id="d2", source="slack"),
    )
    assert len(rows["disagreements"]) == 1
    node = rows["disagreements"][0]
    assert node["key"] == loader._disagreement_key(
        "ENG-4821", subject_key(node["subject"]), "limit"
    )


def test_every_side_is_wired_to_the_disagreement():
    rows = build(
        claim("20%", doc_id="d1", source="confluence"),
        claim("30%", doc_id="d2", source="slack"),
        claim("40%", doc_id="d3", source="jira"),
    )
    node = rows["disagreements"][0]
    over = [row for row in rows["over"] if row["src"] == node["id"]]
    assert len(over) == 3
    assert {row["side"] for row in over} == {0, 1, 2}


def test_generation_is_stamped_on_everything_readable():
    """Nothing in this graph can be deleted, so a reload sits alongside the
    previous one. The stamp is what makes a map current rather than
    cumulative."""
    rows = build(
        claim("20%", doc_id="d1", source="confluence"),
        claim("30%", doc_id="d2", source="slack"),
    )
    assert {row["gen"] for row in rows["claims"]} == {"gen1"}
    assert {row["gen"] for row in rows["disagreements"]} == {"gen1"}


# --- dates ------------------------------------------------------------------


def test_rfc2822_mail_dates_are_read():
    """Gmail is 121,390 of the 511,962 documents and carries RFC 2822 dates.
    Read as an ISO prefix they yield `Tue, 10 Ju`, which sorts as nothing --
    so a quarter of the corpus arbitrated as undated."""
    assert normalise_date("Tue, 10 Jun 2025 09:12:00 -0700") == "2025-06-10"
    assert normalise_date("Wed, 1 Feb 2026 00:00:00 +0000") == "2026-02-01"


def test_iso_dates_pass_through_and_junk_becomes_empty():
    assert normalise_date("2026-02-03") == "2026-02-03"
    assert normalise_date("Wed, Feb 1") == ""
    assert normalise_date("") == ""


def test_undated_claims_never_supersede():
    """An undated claim is neither recent nor stale, and a chain ordered by a
    date nobody recorded is not a history."""
    rows = build(
        claim("open", predicate="status", subject="ENG-4821",
              doc_id="d1", source="slack", date=""),
        claim("shipped", predicate="status", subject="ENG-4821",
              doc_id="d2", source="slack", date=""),
    )
    assert rows["supersedes"] == []


# --- selecting what to read -------------------------------------------------


def test_documents_are_spread_across_tools_rather_than_taken_by_rank():
    """Slack is 56% of the corpus, so a budget filled by score alone is a
    budget spent inside one tool, and one tool cannot disagree with itself
    across tools."""

    rows = [
        {"doc_id": f"slack{i}", "source": "slack", "date": "2026-01-01"}
        for i in range(30)
    ] + [
        {"doc_id": "conf1", "source": "confluence", "date": "2026-01-01"},
        {"doc_id": "jira1", "source": "jira", "date": "2026-01-01"},
    ]

    class FakeRecall:
        class conn:
            @staticmethod
            def execute(sql, params):
                return SimpleNamespace(fetchall=lambda: rows)

        @staticmethod
        def get_many(ids):
            return list(ids)

    picked = loader.documents_for(FakeRecall(), "ENG-4821")
    assert "conf1" in picked and "jira1" in picked
    assert sum(1 for d in picked if d.startswith("slack")) <= loader.DOCS_PER_SOURCE


def test_duplicate_edges_in_one_batch_are_collapsed():
    """Two claims reaching the same document would otherwise write the edge
    twice, which the engine accepts and which makes every count wrong."""
    rows = [
        {"src": 1, "dst": 2},
        {"src": 1, "dst": 2},
        {"src": 1, "dst": 3},
    ]
    assert len(loader._dedupe(rows)) == 2


# --- values that only look like they disagree --------------------------------


def test_a_longer_restatement_is_the_same_position():
    """Extraction returns what the document said, and a document says
    `resolved` in one place and `SUP-3812 is resolved` in another. Two
    positions where the corpus has one is the most damaging thing this can do:
    everything downstream then explains, with a rationale, why it picked a
    side of an argument nobody is having."""
    rows = build(
        claim("resolved", predicate="status", subject="SUP-3812",
              doc_id="d1", source="google_drive"),
        claim("SUP-3812 (transient p95 spike) is resolved", predicate="status",
              subject="SUP-3812", doc_id="d2", source="github"),
    )
    assert rows["disagreements"] == []


def test_negation_is_never_folded_away():
    """`not resolved` contains `resolved`, and collapsing them would hide the
    one disagreement that matters most."""
    rows = build(
        claim("resolved", predicate="status", subject="SUP-3812", doc_id="d1"),
        claim("not resolved", predicate="status", subject="SUP-3812",
              doc_id="d2", source="slack"),
    )
    assert len(rows["disagreements"]) == 1


def test_numbers_are_never_folded_into_longer_numbers():
    """`20` sits inside `120`, and numbers are exactly where real
    disagreements live."""
    rows = build(
        claim("20", doc_id="d1", source="confluence"),
        claim("120", doc_id="d2", source="slack"),
    )
    assert len(rows["disagreements"]) == 1


def test_folding_keeps_the_shortest_form_as_the_value():
    """`resolved` is a status; `resolved after identifying a spike` is a status
    plus a sentence about why."""
    rows = build(
        claim("resolved after identifying a transient spike", predicate="status",
              subject="SUP-3812", doc_id="d1", source="jira"),
        claim("resolved", predicate="status", subject="SUP-3812",
              doc_id="d2", source="github"),
        claim("opened", predicate="status", subject="SUP-3812",
              doc_id="d3", source="hubspot"),
    )
    assert len(rows["disagreements"]) == 1
    node = rows["disagreements"][0]
    # Two sides, not three: the two ways of saying resolved are one position.
    assert node["sides"] == 2


def test_a_decision_that_cannot_be_explained_is_not_presented_as_one():
    """The arithmetic can separate two claims by more than the margin while no
    individual signal is nameable. `accepted — no signal separated these
    claims` is a verdict contradicting its own reason, so it refuses instead."""
    from glasshouse.trust import NO_SIGNAL, arbitrate

    result = arbitrate([
        claim("Landed PR + hotfix", predicate="status", subject="PR-28644",
              doc_id="d1", source="linear", date="2025-03-22"),
        claim("Approved", predicate="status", subject="PR-28644",
              doc_id="d2", source="github", date="2025-03-22"),
    ])
    for conflict in result.conflicts:
        assert not (conflict.decided and conflict.rationale == NO_SIGNAL)


def test_an_undated_rival_is_named_as_the_reason_it_lost():
    """An undated claim is scored as neither current nor stale, which costs it
    trust against a dated one. That gap decided plenty of conflicts while the
    rationale said nothing had separated them."""
    from glasshouse.trust import arbitrate

    result = arbitrate([
        claim("Landed PR + hotfix", predicate="status", subject="PR-28644",
              doc_id="d1", source="linear", date="2025-03-22"),
        claim("Approved", predicate="status", subject="PR-28644",
              doc_id="d2", source="github", date=""),
    ])
    conflict = result.conflicts[0]
    if conflict.decided:
        assert "no date" in conflict.rationale


def test_the_graph_groups_sides_the_way_arbitration_decided_them():
    """The loader used to re-bucket values with its own copy of the grouping
    rule and skip the fold, so `liam` and `liam + maria` were written as two
    opposing sides with a CONTRADICTS edge between them — while arbitration had
    already folded them into one position. The graph then recorded a conflict
    its own verdict did not believe in."""
    rows = build(
        claim("liam", predicate="owner", subject="ENG-4824",
              doc_id="d1", source="slack"),
        claim("liam + maria", predicate="owner", subject="ENG-4824",
              doc_id="d1", source="slack"),
        claim("Sofia/Lena", predicate="owner", subject="ENG-4824",
              doc_id="d2", source="google_drive"),
    )
    node = rows["disagreements"][0]
    assert node["sides"] == 2, "liam and liam + maria are one position"
    # And nothing may contradict a claim on its own side.
    sides = {}
    for row in rows["over"]:
        sides[row["dst"]] = row["side"]
    for row in rows["contradicts"]:
        assert sides[row["src"]] != sides[row["dst"]]


def test_arbitration_does_not_depend_on_the_order_claims_arrive_in():
    """The same claims must arbitrate the same way however retrieval ordered
    them.

    Two values extracted from the *same* document score identical trust and
    carry the same doc id, so `merged` and `done` tied and the sort fell
    through to dict insertion order — which is arrival order. No disagreement
    ever flipped between settled and refused, but the candidate shown for a
    refused one changed between runs, and a panel that renames its own
    candidate on reload cannot be told apart from one that is guessing.
    """
    import random

    from glasshouse.trust import arbitrate, subject_key

    claims = [
        claim("merged", predicate="status", subject="PR-511",
              doc_id="d1", source="github", date="2026-02-01"),
        claim("done", predicate="status", subject="PR-511",
              doc_id="d1", source="github", date="2026-02-01"),
        claim("in review", predicate="status", subject="PR-511",
              doc_id="d2", source="linear", date="2026-02-01"),
        claim("20%", doc_id="d3", source="confluence"),
        claim("30%", doc_id="d4", source="slack"),
    ]

    def snapshot(rows):
        return {
            (c.scope, subject_key(c.subject), c.predicate): (
                c.winner.object_value,
                c.decided,
                c.rationale,
                tuple(loser.object_value for loser in c.losers),
            )
            for c in arbitrate(rows).conflicts
        }

    reference = snapshot(claims)
    assert reference
    for seed in range(8):
        shuffled = claims[:]
        random.Random(seed).shuffle(shuffled)
        assert snapshot(shuffled) == reference
