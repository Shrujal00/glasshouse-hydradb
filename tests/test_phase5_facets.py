"""The structure around a document was normalized and then thrown away.

Every record in `data/normalized` carries the folder, channel, speaker,
attendee and header fields the metadata questions actually key on -- "in the
internal customer success and support knowledge space" is
`folders=['customer-success-and-support']`, and the FTS index stores neither
that nor anything derived from it. These tests pin the two things a facet store
has to get right: finding the container a question paraphrases, and refusing to
open a huge generic container just because the question said its one word.
"""

from __future__ import annotations

import json
from pathlib import Path

from glasshouse.facets import FacetStore


def write_normalized(root: Path, source: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{source}.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"source": source, **row}) + "\n")


def record(doc_id: str, **fields) -> dict:
    base = {
        "doc_id": doc_id,
        "title": f"Note {doc_id}",
        "body": "body text",
        "slug": f"slug-{doc_id}",
        "folders": [],
        "channels": [],
        "speakers": [],
        "attendees": [],
        "named_emails": [],
        "headers": {},
        "ticket_key": None,
        "date": None,
        "thread_id": None,
    }
    base.update(fields)
    return base


def build_store(tmp_path: Path) -> FacetStore:
    normalized = tmp_path / "normalized"
    write_normalized(
        normalized,
        "confluence",
        [
            record("c1", folders=["customer-success-and-support", "integration-guides"]),
            record("c2", folders=["customer-success-and-support"]),
            record("c3", folders=["customer-success"]),
            record("c6", folders=["customer-success"]),
            # `engineering` is the shape that must never open: one word, and it
            # holds 21,841 documents in the real corpus.
            *[record(f"e{i}", folders=["engineering"]) for i in range(9)],
            record("c4", folders=["eng-serving-runtime"]),
            record("c5", folders=["eng-serving-runtime"]),
            record("c7", folders=["eng-serving-runtime"]),
        ],
    )
    write_normalized(
        normalized,
        "google_drive",
        [
            record("g1", folders=["shared_drives", "eng-serving-runtime"]),
            record("g2", folders=["shared_drives", "eng-serving-runtime"]),
        ],
    )
    write_normalized(
        normalized,
        "linear",
        [
            record("l1", folders=["internal-support"], ticket_key="ENG-20491"),
            record("l2", folders=["internal-support"]),
        ],
    )
    write_normalized(
        normalized,
        "slack",
        [
            record(
                "s1",
                folders=["incidents"],
                channels=["incidents"],
                speakers=["maya", "sam"],
                thread_id="2151234567",
            ),
            *[
                record(f"s{i}", folders=["customer-success"], channels=["customer-success"])
                for i in range(2, 5)
            ],
        ],
    )
    write_normalized(
        normalized,
        "gmail",
        [
            record(
                "m1",
                folders=["aditya_rao"],
                named_emails=[
                    {"name": "Tom Nguyen", "email": "tom.nguyen@medispec.com"},
                    {"name": "Tom Nguyen", "email": "tom.nguyen@medispec.com"},
                ],
                headers={
                    "from": "Tom Nguyen <tom.nguyen@medispec.com>",
                    "to": "Aditya Rao <aditya_rao@redwood.ai>",
                    "subject": "MAP kickoff - cost assumptions",
                    "date": "Tue, 10 Jun 2025 09:12:00 -0700",
                },
                date="Tue, 10 Jun 2025 09:12:00 -0700",
            ),
            record("m2", folders=["aditya_rao"]),
        ],
    )
    write_normalized(
        normalized,
        "fireflies",
        [
            record(
                "f1",
                folders=["customer-success"],
                speakers=["Jordan", "Amir"],
                attendees=["Priya Nair", "Daniel Cho"],
                date="2026-08-18",
            ),
            record("f2", folders=["customer-success"]),
        ],
    )

    store = FacetStore(tmp_path / "facets.sqlite3")
    store.build(
        ["confluence", "google_drive", "linear", "slack", "gmail", "fireflies"],
        normalized=normalized,
    )
    return store


def test_build_counts_every_document_and_its_containers(tmp_path):
    store = build_store(tmp_path)
    counts = store.build(
        ["confluence"], normalized=tmp_path / "normalized"
    )  # restartable: a second pass over one source must not double-count
    assert counts["confluence"] == 16

    containers = {c.key: c for c in store.all_containers()}
    assert containers["confluence:folder:customer-success-and-support"].documents == 2
    assert containers["confluence:folder:customer-success"].documents == 2
    assert containers["confluence:folder:engineering"].documents == 9
    assert containers["slack:channel:incidents"].kind == "channel"
    assert containers["slack:folder:incidents"].kind == "folder"


def test_containers_named_finds_a_paraphrased_space(tmp_path):
    store = build_store(tmp_path)
    found = store.containers_named(
        "In the internal customer success and support knowledge space, what is the "
        "escalation path?"
    )
    assert found, "the paraphrased folder name was not matched at all"
    assert found[0].name == "customer-success-and-support"


def test_containers_named_prefers_the_more_specific_container(tmp_path):
    """`customer-success` matches the paraphrase too, and is the smaller
    container -- but it explains less of the question, so it must not win."""
    store = build_store(tmp_path)
    names = [c.name for c in store.containers_named(
        "in the internal customer success and support knowledge space"
    )]
    assert names.index("customer-success-and-support") < names.index("customer-success")


def test_containers_named_matches_a_hyphenated_form_verbatim(tmp_path):
    store = build_store(tmp_path)
    found = store.containers_named("the customer-success channel")
    assert {c.name for c in found} == {"customer-success"}
    assert store.containers_named("the internal-support project")[0].name == (
        "internal-support"
    )


def test_containers_named_follows_the_cue_word_the_question_used(tmp_path):
    """`customer-success` is a Slack channel, a Confluence folder and a
    Fireflies folder at once. Size alone would hand back the Confluence one for
    a question that said "channel", which scopes to the wrong source."""
    store = build_store(tmp_path)
    found = store.containers_named("in the customer-success channel, who paged first?")
    assert (found[0].source, found[0].kind) == ("slack", "channel")


def test_containers_named_opens_a_mailbox_only_for_a_mail_question(tmp_path):
    """A gmail folder is named after whoever owns the mailbox, so a question
    that merely mentions a person must not scope to everything they ever
    received -- 5,862 documents for the busiest mailbox in the corpus."""
    store = build_store(tmp_path)
    assert store.containers_named("what did Aditya Rao decide about the pricing tier?") == []
    found = store.containers_named("in the Aditya Rao inbox, which email named the attachment?")
    assert found[0].key == "gmail:folder:aditya_rao"


def test_containers_named_ignores_a_generic_one_token_name(tmp_path):
    """A container whose whole name is one ordinary word cannot be opened by a
    question that happens to use that word; `engineering` would otherwise scope
    retrieval to 21,841 documents for any question mentioning engineering."""
    store = build_store(tmp_path)
    found = store.containers_named("how did the engineering team pick the retry budget?")
    assert [c.name for c in found] == []


def test_containers_named_breaks_ties_toward_the_smaller_container(tmp_path):
    store = build_store(tmp_path)
    found = store.containers_named("what shipped in the eng-serving-runtime team drive?")
    names = [(c.source, c.documents) for c in found if c.name == "eng-serving-runtime"]
    assert names == [("google_drive", 2), ("confluence", 3)]


def test_containers_named_respects_the_limit(tmp_path):
    store = build_store(tmp_path)
    assert len(store.containers_named("customer success and support", limit=1)) == 1


def test_source_hint_routes_a_repo_question_and_refuses_a_ticket_one(tmp_path):
    store = build_store(tmp_path)
    assert store.source_hint("which repo holds the eval harness?") == "github"
    assert store.source_hint("what does the onboarding runbook page say?") == "confluence"
    assert store.source_hint("what Slack channel hosts that discussion?") == "slack"
    assert store.source_hint("who was the internal organizer of the sync call?") == (
        "fireflies"
    )
    # linear and jira both hold tickets, so a guess here is a coin flip.
    assert store.source_hint("which ticket tracks the regression?") is None
    # Two families firing is the same problem: the question does not say which.
    assert store.source_hint("which repo does the runbook page name?") is None
    assert store.source_hint("who decided this?") is None


def test_facets_for_round_trips_headers_speakers_and_containers(tmp_path):
    store = build_store(tmp_path)
    found = store.facets_for(["m1", "f1", "nonexistent"])
    assert set(found) == {"m1", "f1"}

    mail = found["m1"]
    assert mail.source == "gmail"
    assert mail.headers["subject"] == "MAP kickoff - cost assumptions"
    assert mail.participants == (("Tom Nguyen", "tom.nguyen@medispec.com"),)
    assert "aditya_rao" in mail.containers

    call = found["f1"]
    assert call.speakers == ("Jordan", "Amir")
    assert call.attendees == ("Priya Nair", "Daniel Cho")
    assert call.date == "2026-08-18"
    assert "customer-success" in call.containers


def test_card_states_the_fields_the_question_keys_on(tmp_path):
    store = build_store(tmp_path)
    card = store.facets_for(["m1"])["m1"].card()
    assert "gmail" in card
    assert "Tom Nguyen <tom.nguyen@medispec.com>" in card
    assert "aditya_rao" in card
    assert store.facets_for(["l1"])["l1"].card().count("ENG-20491") == 1


def test_documents_in_respects_the_cap_and_takes_the_cheapest_first(tmp_path):
    store = build_store(tmp_path)
    keys = [
        "confluence:folder:engineering",
        "confluence:folder:customer-success-and-support",
    ]
    assert store.documents_in(keys, limit=3) == ["c1", "c2", "e0"]
    assert len(store.documents_in(keys, limit=5)) == 5
    assert store.documents_in(keys, limit=0) == []
    assert store.documents_in([], limit=5) == []


def test_documents_in_deduplicates_across_overlapping_containers(tmp_path):
    store = build_store(tmp_path)
    keys = ["google_drive:folder:eng-serving-runtime", "google_drive:folder:shared_drives"]
    assert store.documents_in(keys, limit=5) == ["g1", "g2"]
