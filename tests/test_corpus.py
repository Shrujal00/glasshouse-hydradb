"""Parser tests.

Parsing runs unattended over half a million files, so a silent regression here
would quietly corrupt entity resolution downstream. These cases are taken from
real corpus documents.
"""

from pathlib import Path

import pytest

from glasshouse.corpus import Document, parse_document, parse_filename


def write(tmp_path: Path, source: str, name: str, text: str) -> Path:
    d = tmp_path / source
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_filename_extracts_id_and_slug():
    doc_id, slug = parse_filename("dsid_0447408b8e874f2b__gha-eval-smoke.txt")
    assert doc_id == "dsid_0447408b8e874f2b"
    assert slug == "gha-eval-smoke"


def test_parse_filename_falls_back_for_unexpected_names():
    doc_id, slug = parse_filename("notes.txt")
    assert doc_id is None
    assert slug == "notes"


def test_slack_extracts_channel_and_speakers(tmp_path):
    p = write(
        tmp_path,
        "slack",
        "dsid_abc123__2151234567-gha-eval-smoke-runner-flapping.txt",
        "eng-ml\n\n"
        "maya: heads up — PR gating has been flapping\n"
        "sam: ugh, saw those. failing on the quantized smoke step\n"
        "maya: proposal: add a lightweight eval-smoke action\n",
    )
    doc = parse_document(p, "slack", tmp_path)

    assert doc.doc_id == "dsid_abc123"
    assert doc.speakers == ["maya", "sam"]  # deduped, order preserved
    assert "eng-ml" in doc.channels
    assert doc.thread_id == "2151234567"


def test_slack_speaker_regex_ignores_prose_colons(tmp_path):
    p = write(
        tmp_path,
        "slack",
        "dsid_x__1234567-notes.txt",
        "eng-ml\n\nmaya: see this\nNote: this is not a speaker\n",
    )
    doc = parse_document(p, "slack", tmp_path)
    assert doc.speakers == ["maya"]


def test_gmail_extracts_headers_and_named_emails(tmp_path):
    p = write(
        tmp_path,
        "gmail",
        "dsid_1b3840__20250514-private-upgrade-beta.txt",
        "Private upgrades: beta plan\n\n"
        "From: Vivek Kulkarni <vivek_kulkarni@redwoodinference.com>\n"
        "To: Connor O'Brien <connor_obrien@redwoodinference.com>\n"
        "Date: Wed, May 14, 2025 at 9:12 AM PT\n"
        "Subject: Private upgrades\n\n"
        "Connor — I'm proposing we start the beta.\n",
    )
    doc = parse_document(p, "gmail", tmp_path)

    assert doc.headers["from"].startswith("Vivek Kulkarni")
    assert "vivek_kulkarni@redwoodinference.com" in doc.emails
    # name<->email pairs are the strongest entity-resolution signal we get
    names = {ne["name"] for ne in doc.named_emails}
    assert "Vivek Kulkarni" in names
    assert "Connor O'Brien" in names
    # headers must be stripped from the body
    assert doc.body.startswith("Connor —")


def test_ticket_key_parsed_from_slug(tmp_path):
    p = write(tmp_path, "jira", "dsid_000d__SUP-359481-unexpected-5xx.txt", "Title\n\nbody")
    assert parse_document(p, "jira", tmp_path).ticket_key == "SUP-359481"


def test_folders_captured_from_path(tmp_path):
    p = write(
        tmp_path / "google_drive", "shared_drives", "dsid_015b__utilization.txt", "T\n\nb"
    )
    doc = parse_document(p, "google_drive", tmp_path)
    assert doc.folders == ["shared_drives"]


def test_literal_backslash_n_is_unescaped(tmp_path):
    p = write(tmp_path, "confluence", "dsid_061f__field-ops.txt", "Title\n\nOverview:\\n\\nLine two")
    doc = parse_document(p, "confluence", tmp_path)
    assert "\\n" not in doc.body
    assert "Line two" in doc.body


def test_date_from_slug(tmp_path):
    p = write(tmp_path, "fireflies", "dsid_05fa__2025-06-12-security-review.txt", "T\n\nb")
    assert parse_document(p, "fireflies", tmp_path).date == "2025-06-12"


def test_text_property_joins_title_and_body(tmp_path):
    p = write(tmp_path, "github", "dsid_006__pr-29012-cost-pilot.txt", "cost-pilot: batching\n\nMotivation: operators")
    doc = parse_document(p, "github", tmp_path)
    assert doc.text.startswith("cost-pilot: batching")
    assert "Motivation" in doc.text


def test_empty_file_does_not_crash(tmp_path):
    p = write(tmp_path, "jira", "dsid_empty__nothing.txt", "")
    doc = parse_document(p, "jira", tmp_path)
    assert doc.title == ""
    assert doc.body == ""


FIREFLIES = """Security review - Private upgrades (Cascade)

Cascade walked through audit requirements.
Naomi Feldman - Send audit log event list to Cascade. Due: 2025-06-18
Meeting header
Date: 2025-06-12
Attendees (Redwood): Markus Klein, Naomi Feldman, Irene Choi
Attendees (Cascade Financial Group): Priya Nair, Daniel Cho

---

[00:00] Markus Klein: Cool, okay, thanks everyone for joining.
[00:13] Priya Nair: Thanks Markus. We want to go deep on audit logging.
[00:24] Markus Klein: Totally. Quick intros on our side.
"""


def test_fireflies_extracts_full_name_speakers(tmp_path):
    p = write(tmp_path, "fireflies", "dsid_05fa__security-review.txt", FIREFLIES)
    doc = parse_document(p, "fireflies", tmp_path)
    # timestamped turns give full names, deduped
    assert "Markus Klein" in doc.speakers
    assert "Priya Nair" in doc.speakers
    assert doc.speakers.count("Markus Klein") == 1


def test_fireflies_action_item_owners_become_speakers(tmp_path):
    p = write(tmp_path, "fireflies", "dsid_05fa__security-review.txt", FIREFLIES)
    doc = parse_document(p, "fireflies", tmp_path)
    # Naomi never speaks but owns an action item
    assert "Naomi Feldman" in doc.speakers


def test_fireflies_attendees_carry_org_affiliation(tmp_path):
    p = write(tmp_path, "fireflies", "dsid_05fa__security-review.txt", FIREFLIES)
    doc = parse_document(p, "fireflies", tmp_path)
    by_name = {a["name"]: a["org"] for a in doc.attendees}
    assert by_name["Markus Klein"] == "Redwood"
    assert by_name["Priya Nair"] == "Cascade Financial Group"
    assert len(doc.attendees) == 5


def test_fireflies_date_from_meeting_header(tmp_path):
    p = write(tmp_path, "fireflies", "dsid_05fa__security-review.txt", FIREFLIES)
    assert parse_document(p, "fireflies", tmp_path).date == "2025-06-12"
