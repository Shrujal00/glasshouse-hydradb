"""Facets — the structure around a document, which retrieval had been discarding.

Every record in `data/normalized` already carries the folder it lives in, the
channel it was posted to, who spoke, who attended, and the mail headers. None of
it reaches the index: `docs` stores `title` and `body`, and `source`, `date` and
`ticket_key` are UNINDEXED. The metadata questions key on exactly the fields we
dropped -- "in the internal customer success and support knowledge space" is
`folders=['customer-success-and-support']`, and "what Slack channel hosts the
discussion" has its answer in `channels` and nowhere in any document's text.

So this is a third entrance to retrieval, keyed on where a document sits rather
than on who it mentions: a side table of containers, a way to recognise the
container a question paraphrases, and a card of the fields so the model can read
what the index never showed it.

Precision is the whole game in `containers_named`. A container is a retrieval
scope, and `engineering` holds 21,841 documents -- opening it because a question
said "engineering" is worse than not matching at all, because it displaces the
documents plain search would have found. Hence the two-content-token floor: a
container opens when the question names enough of it to be talking about it, not
when the question happens to use one of its words.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .config import NORMALIZED, STATE

FACETS = STATE / "facets.sqlite3"

# Rows per transaction during the build. Half a million documents will not fit
# in memory, and committing per row costs more than parsing the JSON does.
BATCH = 5_000

# Container names are slugs: split them on separators and on case boundaries so
# `eng-serving-runtime`, `shared_drives` and `EvalHarness` all yield the words a
# question would use. The same splitter runs over the question, which is what
# makes a hyphenated form typed by a user match the stored name.
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

# A slug typed verbatim into the question: "the customer-success channel". This
# is the one signal strong enough to open a container on a single content word,
# because nobody writes `internal-support` by accident.
_SLUG = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)+")

# Function words carry no container identity. `customer-success-and-support`
# must match a question that writes "customer success and support" without the
# "and" counting as one of the two content tokens that open the container.
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from how in is it its of on or the "
    "that this to was were what when where which who whose why with".split()
)

# The nouns that signal a container is being referred to rather than naming it:
# "the customer-success *channel*", "the internal support *space*". Dropped from
# both sides of the comparison -- left in the question they would count toward
# the two-token floor and open `support-channel` for any question about support,
# and left in the name they would demand the question repeat the cue word.
_CONTAINER_NOUNS = frozenset(
    "space page channel project drive repo repository folder thread inbox board "
    "workspace directory site team".split()
)

# Question vocabulary that identifies one source. Deliberately small: a wrong
# hint scopes retrieval to the wrong 60,000 documents, so anything that could
# belong to two families is not listed at all, and anything that fires
# alongside another family returns nothing.
_SOURCE_CUES: dict[str, tuple[str, ...]] = {
    "github": ("repo", "repos", "repository", "pull request", "pull-request", "pr", "prs"),
    "confluence": ("space", "spaces", "page", "pages", "playbook", "playbooks",
                   "runbook", "runbooks", "handbook", "handbooks"),
    "slack": ("slack", "channel", "channels", "thread", "threads"),
    "gmail": ("email", "emails", "e-mail", "inbox", "mailbox"),
    "fireflies": ("call", "calls", "meeting", "meetings", "sync", "transcript",
                  "transcripts", "recorded", "recording"),
    "hubspot": ("account", "accounts", "deal", "deals", "pipeline", "stage",
                "stages", "forecast"),
    "google_drive": ("shared drive", "shared_drives", "drive", "doc", "docs",
                     "spreadsheet", "spreadsheets", "sheet", "sheets"),
}

# linear and jira both hold these, and the question never says which. Firing
# suppresses the hint rather than guessing a source.
_AMBIGUOUS_CUES = ("ticket", "tickets", "issue", "issues", "bug", "bugs")

# A container holding one document is not a scope, it is a document -- and
# `channels` outside slack is not a container at all: corpus.py harvests every
# `#hashtag` written in prose into that field, which is 8,544 of the 10,363
# channels in a 124k-document slice. Matching them by name would have the trace
# say "scoped to the #model-serving channel" about a hashtag one author typed
# once. Both kinds are still stored and still shown on the card -- "what Slack
# channel hosts the discussion" is answered from there -- they are just not
# things a question can open.
_MIN_CONTAINER_DOCS = 2
_NAMEABLE = "(kind = 'folder' OR source = 'slack') AND documents >= ?"

_LIST_IN_CARD = 8       # a busy Slack thread has dozens of speakers
_HEADER_CHARS = 240     # a "to:" line can be a hundred addresses long


def _words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(text)]


def _stem(word: str) -> str:
    """Fold a trailing plural so "channels" in a question matches `channel`.

    Words ending in a doubled or vowel-final `s` keep it: stripping it turns
    `success` into `succes` and `status` into `statu`, which would then fail to
    match the name they came from.
    """
    if len(word) >= 4 and word.endswith("s") and not word.endswith(("ss", "us", "is", "os")):
        return word[:-1]
    return word


def _content_sequence(text: str) -> list[str]:
    """The identifying words of a name or a question, in order, with repeats.

    Order is what makes the contiguity test possible, and the test is what
    separates a question that names a container from one that merely uses its
    words: "the internal customer success and support knowledge space" says
    `customer-success-and-support`, while "what is the default request size
    limit for multipart upload support on the API" does not say `support-api`.
    """
    kept: list[str] = []
    for word in _words(text):
        stem = _stem(word)
        if stem in _STOPWORDS or stem in _CONTAINER_NOUNS or _stem(stem) in _CONTAINER_NOUNS:
            continue
        if len(stem) < 2:
            continue
        kept.append(stem)
    return kept


def _content_tokens(text: str) -> list[str]:
    """The distinct words in a name or a question that identify a container."""
    return list(dict.fromkeys(_content_sequence(text)))


def _spoken_as_a_phrase(question: list[str], name: Sequence[str]) -> bool:
    """Does the question say the container's words together, in one run?

    Scattered across a long question they mean nothing -- with 58,138
    containers, some two-word name matches almost any twenty-word question by
    coincidence, and every one of those coincidences would scope retrieval to
    the wrong folder. Adjacent, they are the question naming a place. Word
    order inside the run is not required, because "customer success and
    support" and "support and customer success" name the same folder.
    """
    width = len(name)
    wanted = set(name)
    return any(
        set(question[start:start + width]) == wanted
        for start in range(len(question) - width + 1)
    )


@dataclass(frozen=True)
class Container:
    """A folder or channel, and how much of the corpus opening it would pull in."""

    key: str
    source: str
    kind: str
    name: str
    documents: int


@dataclass(frozen=True)
class DocumentFacets:
    """What the index knew about a document but never showed anybody."""

    doc_id: str
    source: str
    title: str
    date: str
    ticket_key: str
    thread_id: str
    slug: str
    containers: tuple[str, ...]
    speakers: tuple[str, ...]
    attendees: tuple[str, ...]
    participants: tuple[tuple[str, str], ...]
    headers: dict[str, str]

    def card(self) -> str:
        """The rendering handed to the model alongside the document text.

        Answers like "who was the internal organizer" and "what is the filename
        of the attachment" are stated here and nowhere in the body, so a
        document reaching synthesis without its card cannot be answered from.
        """
        lines: list[str] = [f"source: {self.source}"]
        for label, values in (
            ("containers", self.containers),
            ("speakers", self.speakers),
            ("attendees", self.attendees),
        ):
            if values:
                shown = list(values[:_LIST_IN_CARD])
                if len(values) > _LIST_IN_CARD:
                    shown.append(f"+{len(values) - _LIST_IN_CARD} more")
                lines.append(f"{label}: {', '.join(shown)}")
        if self.participants:
            people = [
                f"{name} <{email}>" if name else email
                for name, email in self.participants[:_LIST_IN_CARD]
            ]
            lines.append(f"participants: {', '.join(people)}")
        for field in ("from", "to", "cc", "subject", "date"):
            value = (self.headers.get(field) or "").strip()
            if value:
                lines.append(f"{field}: {value[:_HEADER_CHARS]}")
        # The header block already carried the date when there is one, so
        # repeating it would spend prompt budget restating a line above it.
        if self.date and not self.headers.get("date"):
            lines.append(f"date: {self.date}")
        if self.ticket_key:
            lines.append(f"ticket: {self.ticket_key}")
        if self.thread_id:
            lines.append(f"thread: {self.thread_id}")
        if self.slug:
            lines.append(f"slug: {self.slug}")
        return "\n".join(lines)


class FacetStore:
    """SQLite side table of document facets and the containers holding them."""

    def __init__(self, path: Path = FACETS) -> None:
        self.path = path
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's connection.

        One store is shared by every request FastAPI answers on its worker
        pool, and a SQLite connection belongs to the thread that opened it --
        handing the first worker's connection to the second raises "SQLite
        objects created in a thread can only be used in that same thread",
        intermittently, depending on which worker took the request.
        """
        local = self.__dict__.setdefault("_local", threading.local())
        conn = getattr(local, "conn", None)
        if conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            local.conn = conn
        return conn

    def close(self) -> None:
        local = self.__dict__.setdefault("_local", threading.local())
        conn = getattr(local, "conn", None)
        if conn is not None:
            conn.close()
            local.conn = None

    # --- build ---------------------------------------------------------------

    def create(self) -> None:
        conn = self.conn
        conn.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE IF NOT EXISTS facet (
                doc_id TEXT PRIMARY KEY,
                source TEXT, title TEXT, date TEXT,
                ticket_key TEXT, thread_id TEXT, slug TEXT,
                payload TEXT
            );
            CREATE TABLE IF NOT EXISTS container (
                key TEXT PRIMARY KEY,
                source TEXT, kind TEXT, name TEXT, documents INTEGER
            );
            CREATE TABLE IF NOT EXISTS container_doc (key TEXT, doc_id TEXT);
            CREATE TABLE IF NOT EXISTS container_token (token TEXT, key TEXT);
            """
        )
        conn.commit()

    def index(self) -> None:
        """Indexes, created after the bulk insert rather than during it.

        `documents_in` reads container_doc by key and `facets_for` reads it
        backwards by doc_id, so both directions need one. Without the token
        index `containers_named` scans all 145k token rows per question.
        """
        self.conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS container_doc_key ON container_doc(key);
            CREATE INDEX IF NOT EXISTS container_doc_doc ON container_doc(doc_id);
            CREATE INDEX IF NOT EXISTS container_token_token ON container_token(token);
            """
        )
        self.conn.commit()

    def build(
        self,
        sources: Sequence[str],
        *,
        normalized: Path | None = None,
        limit: int | None = None,
    ) -> dict[str, int]:
        """Load facets for `sources`, returning documents seen per source.

        One source at a time, each clearing its own rows first, so a build
        killed halfway can simply be re-run and a single source can be rebuilt
        without touching the other eight. Container tallies are held in memory
        across a source -- 58,138 of them corpus-wide, which is nothing -- while
        the document rows, which are not, stream through in batches.
        """
        self.create()
        conn = self.conn
        root = normalized or NORMALIZED
        counts: dict[str, int] = {}

        for source in sources:
            path = root / f"{source}.jsonl"
            if not path.exists():
                continue
            self._forget(source)
            tally: dict[str, list] = {}
            facets: list[tuple] = []
            memberships: list[tuple[str, str]] = []
            seen = 0

            for record in _iter_records(path):
                doc_id = record.get("doc_id") or ""
                if not doc_id:
                    continue
                seen += 1
                facets.append(_facet_row(doc_id, source, record))
                for kind, names in (
                    ("folder", record.get("folders") or []),
                    ("channel", record.get("channels") or []),
                ):
                    for name in _clean_names(names):
                        key = f"{source}:{kind}:{name}"
                        entry = tally.get(key)
                        if entry is None:
                            tally[key] = [source, kind, name, 1]
                        else:
                            entry[3] += 1
                        memberships.append((key, doc_id))
                if len(facets) >= BATCH:
                    self._flush(facets, memberships)
                if limit is not None and seen >= limit:
                    break

            self._flush(facets, memberships)
            conn.executemany(
                "INSERT OR REPLACE INTO container (key, source, kind, name, documents)"
                " VALUES (?, ?, ?, ?, ?)",
                [(key, *entry) for key, entry in tally.items()],
            )
            conn.executemany(
                "INSERT INTO container_token (token, key) VALUES (?, ?)",
                [
                    (token, key)
                    for key, entry in tally.items()
                    if _nameable(entry)
                    for token in _content_tokens(entry[2])
                ],
            )
            conn.commit()
            counts[source] = seen

        self.index()
        return counts

    def _forget(self, source: str) -> None:
        """Drop a source's rows so a rebuild replaces rather than doubles.

        Container keys are prefixed with the source, so GLOB on the indexed key
        column is a range scan rather than a pass over 749k membership rows.
        """
        conn = self.conn
        conn.execute("DELETE FROM facet WHERE source = ?", (source,))
        conn.execute("DELETE FROM container_doc WHERE key GLOB ?", (f"{source}:*",))
        conn.execute("DELETE FROM container_token WHERE key GLOB ?", (f"{source}:*",))
        conn.execute("DELETE FROM container WHERE source = ?", (source,))
        conn.commit()

    def _flush(self, facets: list[tuple], memberships: list[tuple[str, str]]) -> None:
        conn = self.conn
        if facets:
            conn.executemany(
                "INSERT OR REPLACE INTO facet"
                " (doc_id, source, title, date, ticket_key, thread_id, slug, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                facets,
            )
            facets.clear()
        if memberships:
            conn.executemany(
                "INSERT INTO container_doc (key, doc_id) VALUES (?, ?)", memberships
            )
            memberships.clear()
        conn.commit()

    # --- read ----------------------------------------------------------------

    def facets_for(self, doc_ids: Sequence[str]) -> dict[str, DocumentFacets]:
        """Facets for the documents synthesis is about to read.

        One indexed query over the primary key, not a lookup per document: this
        runs on the hot path once per question, after retrieval has already
        picked its twenty.
        """
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        rows = self.conn.execute(
            "SELECT doc_id, source, title, date, ticket_key, thread_id, slug, payload"
            f" FROM facet WHERE doc_id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["doc_id"]: _facets(row) for row in rows}

    def containers_named(self, question: str, limit: int = 4) -> list[Container]:
        """The containers this question is talking about, best first.

        A container qualifies when every content token of its name appears in
        the question -- so `customer-success-and-support` needs "support" and
        `customer-success` does not -- and then when either two of those tokens
        matched or the slug itself was typed out. One ordinary word is never
        enough: `engineering` is 21,841 documents and "the engineering team
        decided" is not a request to read all of them.

        Ranked by how much of the name the question accounted for, so the
        paraphrase of the three-word folder beats the two-word folder nested
        inside it. `customer-success` then exists as a Slack channel, a Drive
        folder and a Confluence space at once, so the next tiebreak is the cue
        word the question already used -- "the customer-success *channel*"
        means the Slack one -- and only then the smaller container, which is
        the cheaper and more specific scope.
        """
        spoken = _content_sequence(question)
        tokens = set(spoken)
        slugs = set(_SLUG.findall(question.lower()))
        if not tokens and not slugs:
            return []

        conn = self.conn
        keys: set[str] = set()
        if tokens:
            placeholders = ", ".join("?" for _ in tokens)
            matched: dict[str, int] = {}
            for row in conn.execute(
                f"SELECT token, key FROM container_token WHERE token IN ({placeholders})",
                list(tokens),
            ):
                matched[row["key"]] = matched.get(row["key"], 0) + 1
            keys.update(key for key, hits in matched.items() if hits >= 2)
        if slugs:
            placeholders = ", ".join("?" for _ in slugs)
            keys.update(
                row["key"]
                for row in conn.execute(
                    f"SELECT key FROM container WHERE name IN ({placeholders})",
                    list(slugs),
                )
            )
        if not keys:
            return []

        hint = self.source_hint(question)
        placeholders = ", ".join("?" for _ in keys)
        found: list[tuple[int, int, int, str, Container]] = []
        for row in conn.execute(
            "SELECT key, source, kind, name, documents FROM container"
            f" WHERE key IN ({placeholders}) AND {_NAMEABLE}",
            [*keys, _MIN_CONTAINER_DOCS],
        ):
            content = _content_tokens(row["name"])
            if not content or not all(token in tokens for token in content):
                continue
            if len(content) < 2 and row["name"] not in slugs:
                continue
            if row["name"] not in slugs and not _spoken_as_a_phrase(spoken, content):
                continue
            # A gmail folder is somebody's mailbox, named after them, so any
            # question that happens to say a full name would otherwise open all
            # 5,862 of that person's mails: "the SMB legaltech prospect owned by
            # Alex Martinez" is a CRM question, not a request to read his inbox.
            if row["source"] == "gmail" and row["kind"] == "folder" and hint != "gmail":
                continue
            found.append(
                (
                    -len(content),
                    0 if row["source"] == hint else 1,
                    int(row["documents"]),
                    row["key"],
                    Container(
                        key=row["key"],
                        source=row["source"],
                        kind=row["kind"],
                        name=row["name"],
                        documents=int(row["documents"]),
                    ),
                )
            )
        found.sort(key=lambda item: item[:4])
        return [container for *_, container in found[:limit]]

    def documents_in(self, keys: Sequence[str], limit: int) -> list[str]:
        """Up to `limit` documents from these containers, cheapest container first.

        The cap is not a nicety. `search_scoped` fetches and scores every id it
        is handed, and one container in this corpus holds 28,999 documents --
        so the small, specific container gets read out in full before the huge
        one contributes anything, and the huge one contributes only what is
        left of the budget.
        """
        if not keys or limit <= 0:
            return []
        conn = self.conn
        placeholders = ", ".join("?" for _ in keys)
        ordered = conn.execute(
            f"SELECT key, documents FROM container WHERE key IN ({placeholders})"
            " ORDER BY documents ASC, key ASC",
            list(dict.fromkeys(keys)),
        ).fetchall()

        picked: dict[str, None] = {}
        for row in ordered:
            if len(picked) >= limit:
                break
            # Streamed rather than LIMITed: overlapping containers would
            # otherwise return fewer than the budget allows after dedup.
            for member in conn.execute(
                "SELECT doc_id FROM container_doc WHERE key = ?", (row["key"],)
            ):
                picked.setdefault(member["doc_id"], None)
                if len(picked) >= limit:
                    break
        return list(picked)

    def all_containers(self, limit: int | None = None) -> list[Container]:
        """Every container, largest first — for inspection and build reporting."""
        sql = "SELECT key, source, kind, name, documents FROM container ORDER BY documents DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [
            Container(
                key=row["key"],
                source=row["source"],
                kind=row["kind"],
                name=row["name"],
                documents=int(row["documents"]),
            )
            for row in self.conn.execute(sql)
        ]

    def counts(self) -> tuple[int, int]:
        """(documents, containers) currently stored."""
        documents = int(self.conn.execute("SELECT count(*) FROM facet").fetchone()[0])
        containers = int(self.conn.execute("SELECT count(*) FROM container").fetchone()[0])
        return documents, containers

    def source_hint(self, question: str) -> str | None:
        """The one source this question is asking about, or None.

        Conservative by construction: two families firing means the question
        did not say, and "ticket" cannot distinguish linear from jira. Scoping
        to the wrong source is worse than not scoping, because it removes the
        expected document from the field entirely rather than merely ranking it
        badly.
        """
        text = question.lower()
        tokens = {_stem(word) for word in _words(text)}

        def fires(cues: Sequence[str]) -> bool:
            for cue in cues:
                if " " in cue or "-" in cue or "_" in cue:
                    if cue in text:
                        return True
                elif _stem(cue) in tokens:
                    return True
            return False

        if fires(_AMBIGUOUS_CUES):
            return None
        hits = [source for source, cues in _SOURCE_CUES.items() if fires(cues)]
        return hits[0] if len(hits) == 1 else None


# --- reranking ---------------------------------------------------------------

# The header fields that identify a document rather than describe it. `cc` is
# left out: a long copy list adds tokens without adding aboutness, and it
# dilutes the overlap fraction that does the ranking.
_RERANK_HEADERS = ("from", "to", "subject", "attachments")

# How much a full metadata match is worth against a perfect BM25 score.
# Measured over 30 `metadata` questions against a 500-document page: the
# expected document reaches the top 20 twelve times unweighted, fifteen at 0.5,
# and seventeen from 1.0 upwards, where it plateaus -- so this is the low end of
# the flat region rather than a tuned peak, and BM25 still breaks ties among
# documents whose metadata matches equally.
RERANK_WEIGHT = 1.0


def facet_tokens(facets: DocumentFacets) -> set[str]:
    """Every stemmed content token a document's recorded metadata contributes."""
    parts: list[str] = [facets.slug, facets.title, facets.ticket_key]
    parts.extend(facets.containers)
    parts.extend(facets.speakers)
    parts.extend(facets.attendees)
    parts.extend(facets.headers.get(field, "") for field in _RERANK_HEADERS)
    tokens: set[str] = set()
    for part in parts:
        if part:
            tokens.update(_stem(word) for word in _content_tokens(str(part)))
    return tokens


def rerank(
    question: str,
    candidates: Sequence,
    facets: dict[str, DocumentFacets],
    *,
    weight: float = RERANK_WEIGHT,
    limit: int | None = None,
) -> list:
    """Reorder a deep page by how much of the question the metadata accounts for.

    BM25 ranks on the prose, and a `metadata` question is largely not about the
    prose: "which published page in the customer success and support space"
    shares almost no vocabulary with the page's body and shares nearly all of it
    with the page's filing details. Measured over 30 such questions, the
    expected document sits past rank 20 but inside rank 500 eight times -- found
    by the search and then discarded by the ranking. This is what pulls those
    back.

    Scores are normalised against the best BM25 score in the page rather than
    used raw, so the blend means the same thing whether the question matched
    thousands of documents strongly or a handful weakly.
    """
    ordered = list(candidates)
    if not ordered:
        return []
    asked = {_stem(word) for word in _content_tokens(question)}
    if not asked:
        return ordered[:limit] if limit else ordered
    best = max((getattr(c, "score", 0.0) for c in ordered), default=0.0) or 1.0
    scored = []
    for position, candidate in enumerate(ordered):
        recorded = facets.get(getattr(candidate, "doc_id", ""))
        overlap = (
            len(asked & facet_tokens(recorded)) / len(asked) if recorded else 0.0
        )
        # `position` keeps the sort total and stable: two documents with equal
        # blended scores keep the order the index gave them.
        scored.append(
            (getattr(candidate, "score", 0.0) / best + weight * overlap, -position, candidate)
        )
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    ranked = [row[2] for row in scored]
    return ranked[:limit] if limit else ranked


# --- normalized records ------------------------------------------------------


def _iter_records(path: Path) -> Iterator[dict]:
    """Stream one normalized shard as dicts.

    `recall.iter_normalized` yields index tuples and drops every list field, so
    the facets are not recoverable from it -- but the streaming shape is the
    same, because gmail.jsonl alone is 1.06 GB and must never be read whole.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _nameable(entry: Sequence) -> bool:
    """Is this tallied container something a question is allowed to open?

    Mirrors `_NAMEABLE`, applied while the counts are still in memory so the
    token table never carries the hashtags in the first place.
    """
    source, kind, _, documents = entry
    return (kind == "folder" or source == "slack") and documents >= _MIN_CONTAINER_DOCS


def _clean_names(names: Sequence) -> list[str]:
    """Container names worth keeping, deduped and lowercased.

    jira writes `channels: ['1123']` -- a bare number is not a channel anyone
    can name in a question, and it would only add a token that matches digits.
    """
    kept: dict[str, None] = {}
    for name in names:
        if not isinstance(name, str):
            continue
        name = name.strip().lower().lstrip("#")
        if len(name) < 2 or name.isdigit():
            continue
        kept.setdefault(name, None)
    return list(kept)


def _participants(named_emails: Sequence) -> list[list[str]]:
    """(name, email) pairs from `named_emails`, deduped.

    A mail thread repeats the same "Aditya Rao <aditya_rao@redwood.ai>" in
    every quoted reply, so the raw field routinely holds one person a dozen
    times and would fill the card with them.
    """
    kept: dict[tuple[str, str], None] = {}
    for entry in named_emails:
        if isinstance(entry, dict):
            pair = ((entry.get("name") or "").strip(), (entry.get("email") or "").strip())
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            pair = (str(entry[0]).strip(), str(entry[1]).strip())
        else:
            continue
        if any(pair):
            kept.setdefault(pair, None)
    return [list(pair) for pair in kept]


def _strings(values: Sequence) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if isinstance(value, str) and value.strip():
            seen.setdefault(value.strip(), None)
    return list(seen)


def _facet_row(doc_id: str, source: str, record: dict) -> tuple:
    containers = _clean_names(record.get("folders") or []) + _clean_names(
        record.get("channels") or []
    )
    headers = record.get("headers") or {}
    payload = {
        "containers": list(dict.fromkeys(containers)),
        "speakers": _strings(record.get("speakers") or []),
        "attendees": _strings(record.get("attendees") or []),
        "participants": _participants(record.get("named_emails") or []),
        "headers": {
            str(k): str(v) for k, v in headers.items() if isinstance(headers, dict) and v
        },
    }
    return (
        doc_id,
        record.get("source") or source,
        record.get("title") or "",
        record.get("date") or "",
        record.get("ticket_key") or "",
        record.get("thread_id") or "",
        record.get("slug") or "",
        json.dumps(payload, ensure_ascii=False),
    )


def _facets(row: sqlite3.Row) -> DocumentFacets:
    payload = json.loads(row["payload"] or "{}")
    return DocumentFacets(
        doc_id=row["doc_id"],
        source=row["source"] or "",
        title=row["title"] or "",
        date=row["date"] or "",
        ticket_key=row["ticket_key"] or "",
        thread_id=row["thread_id"] or "",
        slug=row["slug"] or "",
        containers=tuple(payload.get("containers") or ()),
        speakers=tuple(payload.get("speakers") or ()),
        attendees=tuple(payload.get("attendees") or ()),
        participants=tuple(
            (pair[0], pair[1]) for pair in payload.get("participants") or () if len(pair) == 2
        ),
        headers=dict(payload.get("headers") or {}),
    )
