"""Turning evidence into an answer — the last step, and the one with teeth.

Everything before this narrows half a million documents to a handful and works
out who the people in them are. This writes the sentence, and it is where a
system of this kind usually goes wrong: asked a question the corpus cannot
answer, a language model will produce a confident, fluent, invented answer, and
20 of the benchmark's 500 questions exist purely to catch that.

So the model is given three jobs in a fixed order, and refusing is one of them:

1. Answer only from the supplied documents.
2. Cite which document each claim came from.
3. Say plainly when the documents do not contain the answer — while still
   surfacing what *is* there, because a bare "I don't know" is less useful and
   scores worse than a caveated partial answer.

The resolved identities are handed over with the documents, which is where the
ontology repays itself: the model is told outright that `sam h`, `@soham` and
`S. Ratnaparkhi` are one person, so a question about any of those names can be
answered from a document that uses a different one.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

import ollama

from .config import ADJUDICATION_MODEL, STATE, get

if TYPE_CHECKING:  # the renderer below is duck-typed, so this stays type-only
    from .facets import DocumentFacets
    from .trust import Arbitration

# How much of each document reaches the model. The budget is spent on the
# passages that match the question rather than on the opening characters:
# corpus documents run to 5-7k characters and the sentence that answers the
# question is routinely past the halfway mark, so taking the head silently
# dropped the answer and left behind similar-looking numbers that did not
# answer anything.
DOC_CHARS = 2600
# Six was the budget while retrieval had two entrances. With the container
# scope there are three, and measured over 30 `metadata` questions the
# documents it rescues land outside a six-document context every time: the
# expected document reaches the context 8 times at six documents whether or not
# the container entrance ran, 11 at eight, and 12 at ten against ten without
# it. Ordering makes no difference at any budget -- the constraint is the
# budget itself, so it is the budget that moves.
MAX_DOCS = 10

# The unit passages are scored and spent in. Wide enough that a matched term
# arrives with the sentence it belongs to, narrow enough that several separate
# regions of a long document fit inside one document's allowance.
_PASSAGE_WINDOW = 520

# Slices of the budget reserved for the two ends of a long document, before any
# question matching is spent. The tail is the wider of the two because that is
# where the structural facts are written down: google_drive files close with
# "Last edited notes: - Draft created 2026-11-03 (Maya)", confluence with
# "Revision history - 2025-11-10: Created by Ava Martinez", linear and jira
# with a dated activity log. A question about who created a page shares its
# words with the page's *head*, so window scoring spent the whole allowance up
# there and dropped the answering line sitting at 98% depth -- which is how an
# assignee named "Liam Chen" was lost from a document that stated him plainly.
_HEAD_CHARS = 420
_TAIL_CHARS = 560

# The model is told to open with this exact token when the corpus comes up
# short, so abstention is detectable in code rather than inferred from wording.
NOT_FOUND = "NOT_IN_CORPUS"

_STOPWORDS = frozenset(
    "the and for are was were what which who whom whose when where why how "
    "did does do done should would could will can may might must have has had "
    "that this these those there their them they from with without into onto "
    "about above after again against all any because been before being below "
    "between both during each few more most other over same some such than "
    "then they too under until very you your our its his her not but".split()
)

SYSTEM = """You answer questions about a company's internal documents.

Rules, in order of importance:
1. Use ONLY the documents provided. Never use outside knowledge. Never guess.
2. If no document addresses the substance of the question, begin your reply
   with the exact token NOT_IN_CORPUS, then say in one sentence what is
   missing, then briefly describe what related information the documents DO
   contain.
   Judge that on the substance alone. A question usually carries qualifiers —
   "restricted", "published", "merged", "by Elliot Price" — and a document can
   answer the question while recording none of them. If you find yourself
   writing NOT_IN_CORPUS and then naming the very thing that was asked for as
   "related", you were wrong to abstain: that is the answer. Give it, and say
   which qualifier you could not confirm. Reserve NOT_IN_CORPUS for questions
   whose subject is genuinely absent from every document you were given.
3. Write sentences that stand on their own. Each must carry its own subject
   and conditions, so it is still true and still complete for a reader who
   cannot see the question. "Under 10 minutes" is not an answer; "in staging,
   a verified rollback must recover in under 10 minutes" is the same fact
   written so it survives being quoted on its own. Name the system, the
   environment, the threshold and the units every time: not "a firmware bug"
   but "a bug in edge-sw-7 switch firmware version 1.2.3", and keep the units
   the document used rather than converting them.
4. Answer every part of the question, and give every figure the documents
   record about the thing asked about — a question about a test wants the
   conditions it ran under and all of its results, not only the most alarming
   one. If a document records three reviewers, name three; if you can see only
   two, say that only two are recorded rather than presenting them as the
   whole list.
5. Identify what you answered from: exact title, who wrote or published it,
   which space, folder, channel or project it lives in, its status, its date.
   A question asking "which page" is asking you to identify a page, and a page
   is identified by all of these, not by its title alone. State the ones the
   documents record and pass over the ones they do not.
   A question often takes an attribute for granted — "the page by Elliot
   Price", "the merged pull request". Not finding it recorded is not grounds to
   refuse: answer, and note which detail you could not confirm.
6. Cite the documents you used by their bracketed number, like [2].
7. Never invent a citation. Never cite a document you did not use.
8. Graph connections show only that entities co-occur in a document. They do
   not prove collaboration, ownership, agreement, or responsibility.

Rules 3 to 5 never license a guess. Stating a fact in full means stating what a
document says in full, never filling its gaps: an invented hostname, unit or
author is worse than an incomplete answer.

No preamble and no "based on the documents" — but do restate the subject, which
rule 3 requires. Length follows the question: one complete sentence for a
threshold, several when the question has several parts."""


@dataclass(slots=True)
class Written:
    text: str
    abstained: bool
    cited: list[int]


# Temperature 0 is not enough to make a hosted mixture-of-experts model
# reproducible: which experts route a token depends on the composition of the
# batch it lands in, and that is decided by whatever other traffic the service
# is serving at the time. Measured on `gpt-oss:120b-cloud`: three calls with
# identical input and temperature 0 produced two different answers, and on a
# real question the pipeline flipped between naming a team and declining to
# answer at all. Pinning the seed removes it -- three identical outputs -- and
# a demo that answers differently on the second take is not a demo of a system
# anybody would trust.
OPTIONS = {"temperature": 0, "seed": 11}

# ...and the seed is not enough on a long prompt. Measured on
# `gpt-oss:120b-cloud`: an identical 23,009-character prompt with this exact
# options dict produced three substantively different answers -- one naming the
# Scheduler Team, two naming EngPlatform. The same options on a short prompt
# produce three identical answers, so this is the hosted mixture-of-experts
# routing varying with whatever else the service is batching, and no client
# setting reaches it.
#
# What follows from that is a cache, not a workaround: the same question over
# the same evidence returns the answer it returned the first time. Arbitration
# -- which value is *true* -- is decided in `trust.py` and never varies; it is
# only the prose around it that does, and prose that rewrites itself on every
# reload is indistinguishable to a reader from a system changing its mind.
ANSWERS = STATE / "answers.sqlite3"


class _AnswerCache:
    """Written answers, keyed on the prompt that produced them."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or ANSWERS
        self._local = threading.local()
        # Off under test, and off for anyone who wants to see the model answer
        # fresh. A cache that cannot be turned off hides exactly the
        # variability it exists to absorb.
        self.enabled = get("GLASSHOUSE_ANSWER_CACHE", "1") != "0"

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS answers("
                "key TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            conn.commit()
            self._local.conn = conn
        return conn

    @staticmethod
    def key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> str | None:
        if not self.enabled:
            return None
        try:
            row = self.conn.execute(
                "SELECT text FROM answers WHERE key = ?", (self.key(prompt),)
            ).fetchone()
        except Exception:
            return None
        return row[0] if row else None

    def put(self, prompt: str, text: str) -> None:
        if not self.enabled:
            return
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO answers(key, text) VALUES (?, ?)",
                (self.key(prompt), text),
            )
            self.conn.commit()
        except Exception:
            pass


_answers = _AnswerCache()


def _client() -> ollama.Client:
    key = get("OLLAMA_API_KEY")
    return ollama.Client(
        host=get("OLLAMA_HOST", "https://ollama.com"),
        headers={"Authorization": f"Bearer {key}"} if key else None,
    )


def _ends(body: str, *, head_chars: int, tail_chars: int) -> list[tuple[int, int]]:
    """The reserved opening and closing spans, snapped to word boundaries.

    Both are cut at whitespace so the head does not end and the tail does not
    begin mid-word -- a half word reads as corruption, and the closing blocks
    are exactly where a name we want cited lives.
    """
    spans: list[tuple[int, int]] = []
    if head_chars > 0:
        cut = body.rfind(" ", head_chars // 2, head_chars)
        spans.append((0, cut if cut > 0 else min(head_chars, len(body))))
    if tail_chars > 0:
        begin = max(0, len(body) - tail_chars)
        cut = body.find(" ", begin, begin + tail_chars // 2)
        spans.append((cut + 1 if cut > 0 else begin, len(body)))
    return spans


def _render(body: str, spans: list[tuple[int, int]]) -> str:
    """Stitch the chosen spans back together in document order.

    Spans that touch or overlap are merged first: the head, the tail and a
    matched window can land against each other in a document barely over
    budget, and joining those with an ellipsis would show the reader a break
    that is not there -- or repeat the same sentence twice.
    """
    merged: list[list[int]] = []
    for begin, end in sorted(spans):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([begin, end])
    parts = [body[begin:end].strip() for begin, end in merged]
    joined = " … ".join(part for part in parts if part)
    return "… " + joined if merged and merged[0][0] > 0 else joined


def select_passages(body: str, question: str, budget: int = DOC_CHARS) -> str:
    """The parts of `body` that mention the question's own terms, plus its ends.

    Ranked retrieval decides which documents are worth reading; this decides
    which part of one gets read. The document is scored in fixed windows and
    the budget is spent on the best ones that do not overlap, so a dense
    opening cannot swallow the allowance and hide a later passage -- which is
    exactly how the answer to the burst-credit question went missing.

    Rare terms count for more than common ones. A pool identifier that appears
    twice locates the answer; the word "pool", appearing forty times, does not.

    The first and last few hundred characters are taken before any of that, on
    the measurement above `_HEAD_CHARS`: title and opening attribution at one
    end, the revision or activity log at the other, and neither of them is
    reliably reachable from the question's vocabulary.
    """
    body = body.strip()
    if len(body) <= budget:
        return body

    tail_chars = min(_TAIL_CHARS, budget // 4)
    reserved = _ends(body, head_chars=min(_HEAD_CHARS, budget // 5), tail_chars=tail_chars)
    # No term to match on: spend everything the tail does not need on the head,
    # rather than truncating and losing the closing block with it.
    unmatched = _ends(body, head_chars=budget - tail_chars, tail_chars=tail_chars)

    terms = {t for t in re.findall(r"[a-z0-9][a-z0-9._/-]{2,}", question.lower())}
    terms -= _STOPWORDS
    lowered = body.lower()
    weights = {}
    for term in terms:
        seen = lowered.count(term)
        if seen:
            weights[term] = 1.0 / math.sqrt(seen)
    if not weights:
        return _render(body, unmatched)

    step = _PASSAGE_WINDOW // 2
    scored: list[tuple[float, int]] = []
    for begin in range(0, len(body), step):
        chunk = lowered[begin : begin + _PASSAGE_WINDOW]
        score = sum(weight for term, weight in weights.items() if term in chunk)
        if score:
            scored.append((score, begin))
    if not scored:
        return _render(body, unmatched)

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    picked: list[tuple[int, int]] = list(reserved)
    spent = sum(end - begin for begin, end in picked)
    for _, begin in scored:
        end = min(len(body), begin + _PASSAGE_WINDOW)
        if any(begin < other_end and other_begin < end for other_begin, other_end in picked):
            continue
        if spent + (end - begin) > budget:
            continue
        picked.append((begin, end))
        spent += end - begin
    return _render(body, picked)


# What a source calls the thing a document sits inside. The questions use the
# source's own word -- "in the internal customer success and support knowledge
# space", "what Slack channel hosts the discussion" -- and a card that answered
# both with "folder" makes the model translate before it can match.
_CONTAINER_LABEL = {
    "confluence": "space",
    "slack": "channel",
    "google_drive": "drive folder",
    "gmail": "mailbox folder",
    "linear": "project",
    "jira": "project",
    "github": "repo area",
    "fireflies": "folder",
    "hubspot": "record group",
}

# Mail headers in the order a reader expects them. `attachments` is here
# because "the filename of the attachment" is a benchmark question and the
# filename appears nowhere else in the record.
_HEADER_ORDER = ("from", "to", "cc", "subject", "attachments")

_CARD_NOTE = (
    "Some documents below carry a Metadata block. Those are the document's own "
    "recorded fields, exactly as the source system stored them: which space, "
    "channel, folder or project it lives in, its ticket key and dates, mail "
    "from/to/subject, the people recorded as speaking on a call. They are part "
    "of the document and may be quoted and cited like its text. They say where "
    "a document lives and who is recorded on it — never who owns, approved or "
    "decided anything."
)


def _row(label: str, value: str) -> str:
    return f"  {label}: {value[:200].strip()}"


def metadata_card(facets: "DocumentFacets | None") -> str:
    """A document's structural fields, as a block the model can read and cite.

    Every field here is already in `data/normalized/*.jsonl` and none of it has
    ever reached the model: the FTS index stores title and body, and `source`,
    `date` and `ticket_key` are UNINDEXED. The answers to 100 benchmark
    questions live in these fields — the Slack channel that hosted a
    discussion is in `channels` and nowhere in any sentence — and that category
    scores zero.

    Empty fields are dropped rather than printed blank. A card that is mostly
    "(none)" teaches the model that missing metadata is normal and that the
    filled-in lines are as unreliable as the blank ones.
    """
    if facets is None:
        return ""

    def field(name: str) -> str:
        return str(getattr(facets, name, "") or "").strip()

    def listed(name: str) -> list[str]:
        return [str(v).strip() for v in (getattr(facets, name, ()) or ()) if str(v).strip()]

    rows: list[str] = []
    source = field("source")
    if source:
        rows.append(_row("source", source))

    containers = listed("containers")
    if containers:
        label = _CONTAINER_LABEL.get(source, "folder")
        if source == "slack":
            containers = [c if c.startswith("#") else f"#{c}" for c in containers]
        rows.append(_row(label if len(containers) == 1 else f"{label}s", ", ".join(containers[:4])))

    if ticket := field("ticket_key"):
        rows.append(_row("ticket", ticket))
    if date := field("date"):
        rows.append(_row("date", date))

    headers = getattr(facets, "headers", None) or {}
    # `date` is already its own row; anything the source recorded that we did
    # not anticipate is still shown, because one of those is an attachment
    # filename somewhere.
    keys = [k for k in _HEADER_ORDER if headers.get(k)]
    keys += sorted(k for k in headers if headers.get(k) and k not in _HEADER_ORDER and k != "date")
    for key in keys:
        rows.append(_row(key, str(headers[key])))

    for name in ("speakers", "attendees"):
        if people := listed(name):
            rows.append(_row(name, ", ".join(people[:10])))

    # Only when the mail headers did not already name them, or every gmail card
    # prints the same addresses twice.
    if not any(headers.get(k) for k in ("from", "to")):
        pairs = [p for p in (getattr(facets, "participants", ()) or ()) if p and p[0]]
        if pairs:
            rows.append(_row(
                "participants",
                ", ".join(f"{n} <{e}>" if e else str(n) for n, e in pairs[:8]),
            ))

    if thread := field("thread_id"):
        rows.append(_row("thread", thread))
    if slug := field("slug"):
        rows.append(_row("slug", slug))

    return "Metadata:\n" + "\n".join(rows) if rows else ""


def build_prompt(
    question: str,
    docs: Iterable,
    people: Iterable,
    paths: Iterable[dict] | None = None,
    connected: Iterable | None = None,
    *,
    facets: dict[str, "DocumentFacets"] | None = None,
    arbitration: "Arbitration | None" = None,
) -> str:
    """Lay out the evidence, identities first, graph connections last.

    When `paths` is supplied the model sees the multi-hop connections
    HydraDB found between the people this question reached. A shared document
    establishes co-occurrence only, not collaboration, ownership, or agreement.

    `facets` maps doc_id to that document's recorded metadata, which is printed
    with the document it belongs to rather than in a block of its own -- a card
    away from its text is a card the model cannot attribute or cite.
    """
    lines: list[str] = []

    known = [p for p in people if getattr(p, "alias_count", 1) > 1]
    if known:
        lines.append("Identities already resolved (the same person, written several ways):")
        for p in known[:8]:
            lines.append(f"  {p.name} is also written as: {', '.join(sorted(p.surfaces))}")
        lines.append("")

    cards = facets or {}
    if cards:
        lines.append(_CARD_NOTE)
        lines.append("")

    lines.append("Documents:")
    for i, d in enumerate(list(docs)[:MAX_DOCS], start=1):
        body = select_passages(d.text or "", question, budget=DOC_CHARS)
        lines.append(f"\n[{i}] {d.source} — {d.title or d.doc_id}"
                     + (f" ({d.date})" if d.date else ""))
        card = metadata_card(cards.get(d.doc_id))
        if card:
            lines.append(card)
        lines.append(body)

    if arbitration is not None:
        rendered = arbitration.render()
        if rendered:
            lines.append("\n" + rendered)

    everyone = list(connected or ())
    # Someone appearing across several of the retrieved documents is a
    # stronger signal than someone appearing in one. When nobody recurs the
    # ranking says nothing, but the roster is still worth showing -- these are
    # the people the graph attaches to this evidence at all.
    reached = [c for c in everyone if getattr(c, "documents", 0) > 1] or everyone[:5]
    if reached:
        lines.append(
            "\nPeople HydraDB connects to this evidence, by how many of the "
            "documents above they appear in. This is co-occurrence: it shows "
            "who is present, never who owns or decided anything."
        )
        for c in reached[:8]:
            lines.append(f"  {c.name} — in {c.documents} of these documents")

    path_list = list(paths or ())
    if path_list:
        lines.append("\nGraph co-occurrences found by HydraDB (shared documents only):")
        for p in path_list[:6]:
            lines.append(f"  {p.get('summary', '')}")
            via = p.get("via") or []
            if via:
                lines.append(f"    via: {', '.join(str(v)[:60] for v in via[:3])}")

    lines.append(f"\n\nQuestion: {question}")
    return "\n".join(lines)


# The model reaches for CJK brackets — 【2】 — as readily as ASCII ones, so
# citations are normalised rather than demanded. Insisting on one form in the
# prompt and then silently dropping the other loses every citation the answer
# actually made.
_CITE = re.compile(r"[\[【](\d{1,2})[\]】]")


def normalise_citations(text: str) -> str:
    return _CITE.sub(lambda m: f"[{m.group(1)}]", text)


def _finish(text: str) -> Written:
    text = normalise_citations(text.strip())
    abstained = text.startswith(NOT_FOUND)
    if abstained:
        text = text[len(NOT_FOUND) :].lstrip(" :.—-")
    return Written(
        text=text,
        abstained=abstained,
        cited=sorted({int(n) for n in _CITE.findall(text)}),
    )


def write(
    question: str, docs, people, model: str | None = None, connected=None,
    paths: Iterable[dict] | None = None,
    *,
    facets: dict[str, "DocumentFacets"] | None = None,
    arbitration: "Arbitration | None" = None,
) -> Written:
    """Answer in one shot."""
    prompt = build_prompt(
        question, docs, people, paths=paths, connected=connected,
        facets=facets, arbitration=arbitration,
    )
    cached = _answers.get(prompt)
    if cached is not None:
        return _finish(cached)
    response = _client().chat(
        model=model or ADJUDICATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options=OPTIONS,
    )
    written = response["message"]["content"]
    _answers.put(prompt, written)
    return _finish(written)


def write_streaming(
    question: str, docs, people, model: str | None = None, connected=None,
    paths: Iterable[dict] | None = None,
    *,
    facets: dict[str, "DocumentFacets"] | None = None,
    arbitration: "Arbitration | None" = None,
) -> Iterator[dict]:
    """Answer token by token, so the interface can show it being written.

    Yields `{"chunk": ...}` as text arrives and a final `{"done": Written}`.
    The abstention token is stripped from the visible stream: it is a control
    signal for us, not something to show the reader.
    """
    full: list[str] = []
    pending = ""
    checking_marker = True
    prompt = build_prompt(
        question, docs, people, paths=paths, connected=connected,
        facets=facets, arbitration=arbitration,
    )
    cached = _answers.get(prompt)
    # A cached answer is replayed in pieces rather than handed over whole, so
    # the interface still shows it being written. The alternative -- a
    # paragraph appearing instantly on the second ask and slowly on the first
    # -- looks like two different systems.
    if cached is not None:
        stream = ({"message": {"content": cached[i : i + 24]}}
                  for i in range(0, len(cached), 24))
    else:
        stream = _client().chat(
            model=model or ADJUDICATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options=OPTIONS,
            stream=True,
        )
    for part in stream:
        piece = part.get("message", {}).get("content", "")
        if not piece:
            continue
        full.append(piece)
        # Hold the opening back until it is clear whether it is the abstention
        # marker, so the token never flashes on screen.
        if checking_marker:
            pending += piece
            if NOT_FOUND.startswith(pending):
                continue
            if pending.startswith(NOT_FOUND):
                checking_marker = False
                visible = pending[len(NOT_FOUND) :].lstrip(" :.—-")
                pending = ""
                if visible:
                    yield {"chunk": visible}
                continue
            checking_marker = False
            yield {"chunk": pending}
            pending = ""
            continue
        yield {"chunk": piece}
    if pending:
        yield {"chunk": pending}
    written = "".join(full)
    if cached is None and written.strip():
        _answers.put(prompt, written)
    yield {"done": _finish(written)}
