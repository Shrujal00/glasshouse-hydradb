"""Corpus intake: turn EnterpriseRAG-Bench text files into normalized records.

Every source ships as a .txt whose first line is a title, followed by a body
whose shape depends on the source. The filename and directory path carry
structure too, so we mine those rather than only the contents:

    {source}/{folder...}/dsid_{id}__{slug}.txt

The structural signals extracted here (authors, participants, emails, handles,
mentions, ticket keys, dates) are what entity resolution runs on. None of this
needs an LLM, which is why it can cover the whole corpus cheaply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# dsid_<hex>__<slug>.txt
_FILENAME = re.compile(r"^dsid_([0-9a-f]+)__(.+)\.txt$")

# Ticket-ish prefixes that appear in slugs: PM-772222, SUP-359481, pr-29012
_TICKET = re.compile(r"^([A-Za-z]{2,5})-(\d{3,})")

# Trailing dots/commas belong to the sentence, not the address; without the
# trimming group "a@b.com." and "a@b.com" would resolve to two identities.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*[\w]")

# "Vivek Kulkarni <vivek_kulkarni@redwoodinference.com>"
_NAMED_EMAIL = re.compile(r"([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)+)\s*<([^>]+)>")

# Slack transcript line: "maya: heads up — ..."
_SPEAKER = re.compile(r"^([a-z0-9][\w.\-]{1,24}):\s+(?=\S)")

# Fireflies transcript turn: "[00:13] Priya Nair: Thanks Markus."
_TURN = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]\s+([^:]{2,60}?):\s+(?=\S)")

# Fireflies attendee block: "Attendees (Cascade Financial Group): Priya Nair, Daniel Cho"
_ATTENDEES = re.compile(r"^Attendees\s*\(([^)]+)\)\s*:\s*(.+)$", re.MULTILINE)

# Fireflies action item: "Naomi Feldman - Send audit log event list ... Due: 2025-06-18"
_ACTION = re.compile(r"^([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)+)\s+-\s+(?=\S)", re.MULTILINE)

# "Date: 2025-06-12" inside a meeting header
_META_DATE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)

# @mention and #channel. The lookbehind stops the domain half of an email
# address ("...@redwood.com") from being harvested as a mention.
_MENTION = re.compile(r"(?<![\w.+-])@([\w.\-]{2,32})")

# Automation accounts are not people; they must not enter entity resolution
# as candidate humans.
_BOT = re.compile(r"(?:^|[-_])(?:bot|ci|cd|jenkins|webhook|noreply|no-reply)(?:$|[-_])|^(?:deploy|ops|infra|build|release|alert)[-_]", re.I)

# "Attendees (as recorded):" and friends carry no real organisation.
_NON_ORG = re.compile(r"^(?:as\s+)?(?:recorded|listed|captured|noted|above|below|n/?a|unknown|tbd)$", re.I)
_CHANNEL = re.compile(r"#([\w\-]{2,40})")

_HEADER = re.compile(r"^(From|To|Cc|Bcc|Date|Subject|Attachments):\s*(.*)$")

# Leading date in a slug: 2025-06-12-security-review...
_SLUG_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")

# Slack slugs start with a numeric thread id: 2151234567-gha-eval-...
_SLUG_THREAD = re.compile(r"^(\d{6,})-")


@dataclass
class Document:
    """One normalized corpus document."""

    doc_id: str
    source: str
    title: str
    body: str
    slug: str
    folders: list[str] = field(default_factory=list)
    # structural signals for entity resolution
    emails: list[str] = field(default_factory=list)
    named_emails: list[dict[str, str]] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    # meeting transcripts: full names tied to an organisation, the single
    # strongest signal for separating employees from customer contacts
    attendees: list[dict[str, str]] = field(default_factory=list)
    bots: list[str] = field(default_factory=list)
    ticket_key: str | None = None
    date: str | None = None
    thread_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def text(self) -> str:
        """Title plus body, for embedding and retrieval."""
        return f"{self.title}\n\n{self.body}" if self.title else self.body


def _unescape(text: str) -> str:
    r"""Some sources embed literal \n sequences rather than real newlines."""
    return text.replace("\\n", "\n") if "\\n" in text else text


def is_bot(handle: str) -> bool:
    """True when a handle looks like an automation account rather than a person."""
    return bool(_BOT.search(handle))


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def parse_filename(name: str) -> tuple[str | None, str]:
    """Return (doc_id, slug) from a corpus filename.

    Falls back to the bare stem when the name does not match the dsid_ form,
    so an unexpected file still flows through with a usable identity.
    """
    m = _FILENAME.match(name)
    if not m:
        return None, Path(name).stem
    return f"dsid_{m.group(1)}", m.group(2)


def _split_title(raw: str) -> tuple[str, str]:
    """First non-empty line is the title; the remainder is the body."""
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if line.strip():
            return line.strip(), "\n".join(lines[i + 1 :]).strip()
    return "", ""


def _parse_headers(body: str) -> tuple[dict[str, str], str]:
    """Pull leading RFC-822-ish headers (gmail) off the front of a body."""
    headers: dict[str, str] = {}
    lines = body.split("\n")
    idx = 0
    for i, line in enumerate(lines):
        if not line.strip():
            # blank line only ends the block once we have seen a header
            if headers:
                idx = i + 1
                break
            continue
        m = _HEADER.match(line)
        if m:
            headers[m.group(1).lower()] = m.group(2).strip()
            idx = i + 1
        elif headers:
            idx = i
            break
        else:
            return {}, body
    return headers, "\n".join(lines[idx:]).strip()


def parse_document(path: Path, source: str, root: Path) -> Document:
    """Parse one corpus file into a normalized Document."""
    raw = _unescape(path.read_text(encoding="utf-8", errors="replace"))
    doc_id, slug = parse_filename(path.name)
    title, body = _split_title(raw)

    rel = path.relative_to(root)
    folders = list(rel.parts[1:-1])  # drop the source dir and the filename

    headers, stripped = _parse_headers(body) if source == "gmail" else ({}, body)
    if headers:
        body = stripped

    named = [
        {"name": n.strip(), "email": e.strip().lower()}
        for n, e in _NAMED_EMAIL.findall(raw)
    ]
    emails = _dedupe([e.lower() for e in _EMAIL.findall(raw)])

    speakers: list[str] = []
    channels: list[str] = []
    attendees: list[dict[str, str]] = []
    if source == "slack":
        # For slack the title line is the channel, not a subject.
        if title and " " not in title:
            channels.append(title)
        speakers = _dedupe(
            [m.group(1) for line in body.split("\n") if (m := _SPEAKER.match(line))]
        )
    elif source == "fireflies":
        # Timestamped turns carry full names, unlike slack's bare handles.
        speakers = _dedupe(
            [m.group(1).strip() for line in body.split("\n") if (m := _TURN.match(line))]
        )
        for org, names in _ATTENDEES.findall(body):
            org = org.strip()
            if _NON_ORG.match(org):
                org = ""
            for name in names.split(","):
                if name := name.strip():
                    attendees.append({"name": name, "org": org})
        # Action-item owners are named even when they never speak.
        speakers = _dedupe(speakers + _ACTION.findall(body))

    mentions = _dedupe(_MENTION.findall(raw))[:64]
    bots = [m for m in mentions if is_bot(m)]
    bots += [s for s in speakers if is_bot(s)]
    mentions = [m for m in mentions if not is_bot(m)]
    speakers = [s for s in speakers if not is_bot(s)]

    ticket_key = None
    if t := _TICKET.match(slug):
        ticket_key = f"{t.group(1).upper()}-{t.group(2)}"

    date = (
        d.group(1)
        if (d := _SLUG_DATE.match(slug))
        else (md.group(1) if (md := _META_DATE.search(body)) else headers.get("date"))
    )
    thread_id = th.group(1) if (th := _SLUG_THREAD.match(slug)) else None

    return Document(
        doc_id=doc_id or slug,
        source=source,
        title=title,
        body=body,
        slug=slug,
        folders=folders,
        emails=emails,
        named_emails=named,
        speakers=speakers,
        mentions=mentions,
        channels=_dedupe(channels + _CHANNEL.findall(raw))[:32],
        headers=headers,
        attendees=attendees,
        bots=_dedupe(bots),
        ticket_key=ticket_key,
        date=date,
        thread_id=thread_id,
    )


def parse_document_text(text: str, source: str) -> dict[str, list[str]]:
    """Mine identity surfaces out of raw document text.

    The same signals `parse_document` extracts at intake, but reading a string
    rather than a file, because at question time the document arrives from the
    recall index rather than from disk. Used to find who a retrieved document
    is about without re-reading the corpus.
    """
    raw = _unescape(text)
    emails = _dedupe([e.lower() for e in _EMAIL.findall(raw)])
    names = [n.strip() for n, _ in _NAMED_EMAIL.findall(raw)]

    handles: list[str] = []
    for line in raw.split("\n"):
        if m := _SPEAKER.match(line):
            handles.append(m.group(1))
        elif m := _TURN.match(line):
            names.append(m.group(1).strip())

    for org, people in _ATTENDEES.findall(raw):
        names.extend(p.strip() for p in people.split(",") if p.strip())
    names.extend(_ACTION.findall(raw))
    handles.extend(_MENTION.findall(raw))

    return {
        "emails": [e for e in emails if "@" in e and len(e) < 120],
        "names": _dedupe([n for n in names if n and len(n) < 80]),
        "handles": _dedupe([h for h in handles if not is_bot(h)]),
    }


def iter_source_files(root: Path, source: str):
    """Yield every file under one source directory."""
    yield from sorted((root / source).rglob("*.txt"))


_BODY_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def derive_date(record: dict) -> str:
    """The document's own date, or the latest one written inside it.

    Only Gmail and Fireflies carry a date field: measured across the corpus,
    the other seven sources record one on under 2% of their documents, which is
    285,605 Slack messages and every Jira ticket with no date at all. That is
    not a cosmetic gap. Arbitration ranks competing claims partly on recency,
    and a claim with no date cannot be ranked on it, so a later correction and
    the earlier report it corrects arrive at the adjudicator indistinguishable.

    Linear, Jira, HubSpot and Confluence write their dates into the body
    instead, as dated activity logs and revision histories. The latest of those
    is taken: for a log that grows downwards it is the last entry, which is the
    document's most recent activity and the thing recency is asking about.

    The heuristic is honest about being one. A document whose only date is a
    future deadline gets that deadline, which is wrong for "last updated" and
    right for "due date"; both beat having no date at all. Slack and GitHub
    write no ISO dates, so they keep none rather than being given a guess.
    """
    stated = (record.get("date") or "").strip()
    if stated:
        return stated
    found = _BODY_DATE.findall(record.get("body") or "")
    if not found:
        return ""
    return max("-".join(parts) for parts in found)


SOURCES = (
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
)
