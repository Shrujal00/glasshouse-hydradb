"""Turning the evidence into explicit claims, so something other than the model
can decide which one is true.

Retrieval is not what fails the benchmark's `conflicting_info` questions. Plain
FTS puts the expected document in the synthesis context on 10 of 10 of them,
ranked first or second in 9. The model reads two competing values -- "reserve
30% (previous internal suggestion was 20%)" -- and answers with whichever one
it saw first, because nothing in the prompt tells it which to prefer. Pulling
the values out as records is what makes them arbitrable at all; `trust.py` does
the deciding.

The vocabulary is deliberately five predicates wide. A free-form extractor
would produce `assigned_to`, `assignee`, `responsible party` and `owner` for
one relation and then never group them, so anything outside the five is dropped
and stays where it already was -- in the passage text the model reads anyway.
Nothing is force-normalized into a predicate it does not fit.

Every defence in here exists because this is the step where a system like this
invents data. The prompt states the schema; every field is checked; provenance
is taken from the retrieved document rather than the model's word for it; a
value that does not appear in the text the model was shown is discarded; and a
model that answers in prose yields no claims rather than an exception.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import ollama

from .answer import DOC_CHARS, select_passages
from .config import ADJUDICATION_MODEL, STATE, get

PREDICATES = ("owner", "status", "due_date", "limit", "reports_to")

# Bumped whenever the prompt, the schema or the validation changes: cached rows
# from an older extractor describe a different pipeline and must not be reused.
EXTRACTOR_VERSION = "3"

CLAIMS = STATE / "claims.sqlite3"

# The same ceiling synthesis uses. Extracting from documents the model will
# never see produces claims about evidence the answer cannot cite.
MAX_CLAIM_DOCS = 6

# Long enough for a name, a status phrase or a threshold; short enough that a
# model returning a paragraph as the "value" is rejected rather than stored as
# a fact nobody can group.
MAX_SUBJECT = 120
MAX_VALUE = 200

_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})")
# Mail carries RFC 2822 dates -- `Tue, 10 Jun 2025 09:12:00 -0700` -- and gmail
# is 121,390 of the 511,962 documents. Read as an ISO prefix that yields
# `Tue, 10 Ju`, which is not a date, so every claim from a quarter of the
# corpus arbitrated as undated and displayed as nonsense. Recency is half of
# how a conflict is settled and the whole of what a supersession chain is
# ordered by, so this is not cosmetic.
_RFC2822 = re.compile(r"\b(\d{1,2})\s+([A-Z][a-z]{2})[a-z]*\s+(\d{4})\b")
_MONTHS = {
    m: i
    for i, m in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
        start=1,
    )
}


def normalise_date(raw: str) -> str:
    """`YYYY-MM-DD`, or empty when the string does not carry a whole date.

    Empty rather than a truncation: `trust.py` treats an undated claim as
    neither recent nor stale, which is the right answer for a date nobody
    recorded and the wrong answer for one that was recorded in a format this
    failed to read. Better to have neither than to sort on `Wed, Feb 1`.
    """
    text = (raw or "").strip()
    iso = _DATE.search(text)
    if iso:
        return iso.group(1)
    rfc = _RFC2822.search(text)
    if rfc:
        day, month, year = rfc.groups()
        if month in _MONTHS:
            return f"{year}-{_MONTHS[month]:02d}-{int(day):02d}"
    return ""
_FENCE = re.compile(r"^\s*```(?:json)?|```\s*$")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    subject: str
    predicate: str
    object_value: str
    doc_id: str
    source: str
    asserted_at: str
    extractor_confidence: float
    # What the claim was extracted *within*. Two documents only contradict each
    # other when they are talking about the same thing, and the corpus's own
    # answer to "the same thing" is the work item both documents name. Empty
    # for the query path, where the question itself is the scope and every
    # document was retrieved for it.
    scope: str = ""
    trust: float = 0.0
    status: str = "accepted"  # accepted | disputed | superseded
    rationale: str = ""


SYSTEM = f"""You extract explicit factual assertions from company documents.

Return ONLY a JSON object, with no prose before or after it and no code fence:

{{"claims": [{{"document": <int>, "subject": "<string>", "predicate": "<one of: {' | '.join(PREDICATES)}>", "object_value": "<string copied from the document>", "confidence": <number between 0 and 1>}}]}}

Rules:
1. Use ONLY those five predicates. If an assertion does not fit one of them,
   leave it out entirely. Do not invent a predicate and do not stretch one.
2. Copy object_value verbatim from the document. Never paraphrase it, never
   normalise it, never supply a value the document does not state.
3. "document" is the bracketed number of the document you read it in.
4. One claim per assertion. Do not merge two documents into one claim.
5. confidence is how plainly the document states it: 0.9 when the sentence
   asserts it outright, 0.4 when you are reading between the lines.
6. If the documents assert nothing over these five predicates, return
   {{"claims": []}}. That is a correct answer, not a failure."""

REPAIR = """That was not parseable JSON. Return the same information as a single
JSON object of the exact shape given, and nothing else. If you have no claims,
return {"claims": []}."""


def _client() -> ollama.Client:
    """Kept here rather than imported so tests have one place to stub."""
    key = get("OLLAMA_API_KEY")
    return ollama.Client(
        host=get("OLLAMA_HOST", "https://ollama.com"),
        headers={"Authorization": f"Bearer {key}"} if key else None,
    )


def _flat(text: str) -> str:
    """Casefolded alphanumerics only — the form two spellings of one value
    agree on. `In Review.` and `in review` are the same assertion, and a
    grounding check that says otherwise rejects real claims."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _grounded(value: str, passage: str) -> bool:
    """Whether the value is in the text the model was shown.

    The characteristic invention here is a plausible colleague: asked who owns
    the audit-log shipper, a model that has seen `Jordan Reyes` elsewhere in
    the conversation will answer `Jordan Reyes` from a document that says
    `Priya Nair`. A verbatim value survives this check; a remembered one does
    not. Only the value is checked, not the subject -- a subject the documents
    never mention produces a group that never conflicts with anything, while a
    fabricated value is what would reach the reader as the answer.
    """
    flat = _flat(value)
    return bool(flat) and flat in _flat(passage)


def _claim_id(doc_id: str, predicate: str, subject: str, value: str) -> str:
    seed = f"{doc_id}|{predicate}|{_flat(subject)}|{_flat(value)}"
    return hashlib.sha1(seed.encode()).hexdigest()[:12]


def _confidence(raw: object) -> float | None:
    """Model confidence, or None when the field is not a number in range.

    A missing or nonsense confidence is treated as a malformed claim rather
    than defaulted, because the default would be indistinguishable from a
    model that had actually judged the assertion weak.
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


def _parse(text: str) -> list[dict] | None:
    """The claims array, or None when the output is not JSON at all.

    None is the signal that a repair attempt is worth making; an empty list is
    a model that read the documents and found nothing, which is not an error.
    """
    stripped = _FENCE.sub("", text.strip()).strip()
    begin = stripped.find("{")
    end = stripped.rfind("}")
    if begin == -1 or end <= begin:
        return None
    try:
        loaded = json.loads(stripped[begin : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    claims = loaded.get("claims")
    if not isinstance(claims, list):
        return None
    return [item for item in claims if isinstance(item, dict)]


def _validate(
    item: dict, documents: Sequence, passages: dict[str, str], scope: str = ""
) -> Claim | None:
    """One raw object into a claim, or nothing. Never a partially guessed one."""
    index = item.get("document")
    if isinstance(index, str) and index.strip().isdigit():
        index = int(index)
    if not isinstance(index, int) or not 1 <= index <= len(documents):
        return None
    predicate = item.get("predicate")
    if predicate not in PREDICATES:
        return None
    subject = item.get("subject")
    value = item.get("object_value")
    if not isinstance(subject, str) or not isinstance(value, str):
        return None
    subject, value = subject.strip(), value.strip()
    if not subject or not value or len(subject) > MAX_SUBJECT or len(value) > MAX_VALUE:
        return None
    confidence = _confidence(item.get("confidence"))
    if confidence is None:
        return None
    document = documents[index - 1]
    if not _grounded(value, passages.get(document.doc_id, "")):
        return None
    # Provenance is ours, not the model's. Which document a sentence came from
    # and when it was written are facts retrieval already established, and
    # letting the model restate them is letting it get them wrong.
    asserted_at = getattr(document, "date", "") or ""
    return Claim(
        claim_id=_claim_id(document.doc_id, predicate, subject, value),
        subject=subject,
        predicate=predicate,
        object_value=value,
        doc_id=document.doc_id,
        source=getattr(document, "source", "") or "",
        asserted_at=normalise_date(asserted_at),
        extractor_confidence=confidence,
        scope=scope,
    )


class _Cache:
    """Extraction results keyed by what was extracted from.

    Grading re-asks the same questions and the UI re-asks the same question the
    moment anything downstream changes, and every repeat would otherwise be a
    fresh paid call over the same six documents. The key is the passage text
    plus the extractor version, so a changed prompt or a changed question
    invalidates itself without anyone having to remember to clear anything.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's connection. FastAPI answers on a worker pool and a
        SQLite connection belongs to the thread that opened it, so sharing one
        raises "SQLite objects created in a thread can only be used in that
        same thread" -- intermittently, depending on which worker took the
        request."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS extracted("
                "key TEXT PRIMARY KEY, version TEXT NOT NULL, claims TEXT NOT NULL)"
            )
            conn.commit()
            self._local.conn = conn
        return conn

    def get(self, key: str) -> list[dict] | None:
        row = self.conn.execute(
            "SELECT claims FROM extracted WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, claims: list[Claim]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO extracted(key, version, claims) VALUES (?, ?, ?)",
            (key, EXTRACTOR_VERSION, json.dumps([asdict(c) for c in claims])),
        )
        self.conn.commit()


def _key(doc_id: str, passage: str, scope: str = "") -> str:
    seed = f"{EXTRACTOR_VERSION}|{scope}|{doc_id}|{passage}"
    return hashlib.sha256(seed.encode()).hexdigest()


SCOPED = """
7. Every document below concerns {scope}. Two documents that assert something
   about the same thing must write the subject the same way, or the fact that
   they disagree cannot be found. So name the subject as plainly as the
   documents do, and use the identical wording across documents.
8. Write the subject as exactly "{scope}" ONLY when the assertion is about the
   work item as a whole -- its own status, its own owner, its own due date.
   Anything about a threshold, a component, a person or a sub-task takes that
   thing's name as the subject, never "{scope}". Two different thresholds
   mentioned in one meeting are two subjects, not two competing values of one:
   filing them under the same subject would report the document as
   contradicting itself when it does nothing of the kind."""


def _prompt(
    question: str, documents: Sequence, passages: dict[str, str], scope: str = ""
) -> str:
    lines = [f"Question under investigation: {question}", "", "Documents:"]
    for i, document in enumerate(documents, start=1):
        date = getattr(document, "date", "")
        title = getattr(document, "title", "") or document.doc_id
        lines.append(f"\n[{i}] {document.source} — {title}" + (f" ({date})" if date else ""))
        lines.append(passages.get(document.doc_id, ""))
    return "\n".join(lines)


def _ask(
    question: str,
    documents: Sequence,
    passages: dict[str, str],
    model: str | None,
    scope: str = "",
) -> list[Claim] | None:
    """One batched call, one repair attempt, then give up.

    None means the model never produced parseable JSON, which is different from
    a model that produced JSON with nothing in it: the first must not be cached
    as "this document asserts nothing", the second must.
    """
    client = _client()
    system = SYSTEM + (SCOPED.format(scope=scope) if scope else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _prompt(question, documents, passages, scope)},
    ]
    raw = client.chat(
        model=model or ADJUDICATION_MODEL, messages=messages, options={"temperature": 0}
    )["message"]["content"]
    items = _parse(raw)
    if items is None:
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": REPAIR},
        ]
        raw = client.chat(
            model=model or ADJUDICATION_MODEL, messages=messages, options={"temperature": 0}
        )["message"]["content"]
        items = _parse(raw)
    if items is None:
        return None
    found = [_validate(item, documents, passages, scope) for item in items]
    return [claim for claim in found if claim is not None]


def extract(
    documents: Sequence,
    question: str,
    *,
    model: str | None = None,
    cache_path: Path | None = None,
    scope: str = "",
) -> list[Claim]:
    """Claims over the fixed vocabulary, from the documents synthesis will see.

    The documents are trimmed with the same `select_passages` the answer prompt
    uses, so a claim can never be extracted from text the answer was not shown
    -- otherwise arbitration hands the model a value it cannot find in its own
    evidence and asks it to defend it.

    `scope` names the thing every one of these documents is about — a work item
    key, when the caller is the offline loader walking the corpus. It is
    stamped on every claim and it is what arbitration groups within, so a
    Confluence page and a Slack thread about the same ticket can contradict
    each other while two unrelated tickets that both happen to have an `owner`
    cannot. The query path leaves it empty: there, the question is the scope.

    Never raises. A missing model, a timeout, a corrupt cache and a model
    talking prose all degrade to fewer claims, and fewer claims degrades to the
    behaviour that exists today.
    """
    try:
        docs = [
            d for d in list(documents)[:MAX_CLAIM_DOCS] if getattr(d, "doc_id", None)
        ]
        if not docs:
            return []
        passages = {
            d.doc_id: select_passages(getattr(d, "text", "") or "", question, budget=DOC_CHARS)
            for d in docs
        }
        cache: _Cache | None
        try:
            cache = _Cache(cache_path or CLAIMS)
        except Exception:
            cache = None

        claims: list[Claim] = []
        missing = []
        for document in docs:
            key = _key(document.doc_id, passages[document.doc_id], scope)
            stored = cache.get(key) if cache is not None else None
            if stored is None:
                missing.append(document)
                continue
            claims.extend(Claim(**row) for row in stored)
        if not missing:
            return claims

        fresh = _ask(question, missing, passages, model, scope)
        if fresh is None:
            return claims
        by_document: dict[str, list[Claim]] = {d.doc_id: [] for d in missing}
        for claim in fresh:
            by_document[claim.doc_id].append(claim)
        for document in missing:
            found = by_document[document.doc_id]
            if cache is not None:
                cache.put(
                    _key(document.doc_id, passages[document.doc_id], scope), found
                )
            claims.extend(found)
        return claims
    except Exception:
        return []
