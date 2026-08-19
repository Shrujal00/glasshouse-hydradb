"""Local recall — narrowing half a million documents down to the handful worth reading.

Everything the ontology does depends on getting the right ~20 documents in
front of it, and doing that locally rather than through a hosted service is
what lets the whole system run from `docker compose up` with no account, no
key and no rate limit.

SQLite's FTS5 does the work. It is not a semantic index and makes no pretence
of being one: it matches words. That is enough to be the first stage, because
the second stage is a graph that knows `@soham` and `S. Ratnaparkhi` are the
same person - which is precisely the kind of connection a vector store cannot
make and the reason recall alone was never going to answer these questions.

Ranking is BM25 with the title weighted above the body. In this corpus a Jira
summary or an email subject line routinely states the fact the body only
alludes to, so a title hit is worth more than a body hit.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .config import NORMALIZED, STATE

INDEX_PATH = STATE / "recall.sqlite3"

# FTS5 reads bare punctuation as query syntax, so a question mark or a hyphen
# in a user's question is a parse error rather than a search term.
_TERM = re.compile(r"[A-Za-z0-9_]+(?:[.\-/][A-Za-z0-9_]+)*")

# A term matching more than this share of the corpus tells us nothing about
# which document to read, and costs the most to evaluate. Measured rather than
# listed: a stopword list is a guess about English, while document frequency
# is a fact about this corpus - and in a corpus of API documentation, "request"
# is a stopword while in a corpus of poetry it is not.
MAX_DF_FRACTION = 0.05

# When every term in a question is common, keep this many of the rarest rather
# than searching for nothing.
MIN_TERMS = 4
MAX_SCOPE_TERMS = 6

_QUESTION_TERM = frozenset(
    "a an about are decide decided did do does how is me of please say said tell the was were what when where which who why".split()
)


@dataclass(slots=True)
class Candidate:
    """One retrieved document, with enough of it to extract from."""

    doc_id: str
    source: str
    title: str
    body: str
    date: str
    score: float

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()

    def cite(self) -> str:
        """A short human-readable citation, the form the answer quotes."""
        where = f"{self.source}"
        if self.date:
            where += f", {self.date}"
        return f"{self.title or self.doc_id} ({where})"


def query_terms(question: str) -> list[str]:
    """Turn a natural-language question into candidate FTS5 terms.

    Identifier-ish tokens are kept whole - `audit-log-shipper`, `PM-772222`
    and `p99.9` are the most discriminating things a question can contain, and
    splitting them on punctuation throws that away.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for match in _TERM.finditer(question.lower()):
        term = match.group(0)
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


class LocalRecall:
    """Full-text index over the normalized corpus."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or INDEX_PATH
        self._local = threading.local()
        self._total: int | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's connection.

        A SQLite connection belongs to the thread that opened it, and one
        `LocalRecall` is shared by every request the server answers on its
        worker pool. Handing the first thread's connection to the second
        raises "SQLite objects created in a thread can only be used in that
        same thread" -- intermittently, depending on which worker took the
        request. Reads are the whole workload here, so a connection per
        thread costs nothing and needs no lock. `search_scoped` also builds a
        temporary table, which is per-connection and so stays private too.
        """
        local = self.__dict__.setdefault("_local", threading.local())
        conn = getattr(local, "conn", None)
        if conn is None:
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
        """(Re)create the index.

        `date`, `ticket_key` and `thread_id` stay unindexed - they are for
        filtering and citation, and indexing them would dilute BM25.

        `facets` is the exception, and it is indexed deliberately. The
        normalized records carry the folder, the channel, the speakers and the
        mail headers, and none of it used to reach the index at all: a question
        asking "in the internal customer success and support knowledge space"
        was matched only against prose that never names the space it is filed
        in. It is its own column rather than appended to the body so it can
        carry its own BM25 weight and so a match in it stays attributable.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.conn
        conn.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            DROP TABLE IF EXISTS docs;
            CREATE VIRTUAL TABLE docs USING fts5(
                doc_id UNINDEXED,
                source UNINDEXED,
                title,
                body,
                facets,
                date UNINDEXED,
                ticket_key UNINDEXED,
                thread_id UNINDEXED,
                tokenize = 'porter unicode61'
            );
            CREATE TABLE IF NOT EXISTS term_df (term TEXT PRIMARY KEY, df INTEGER);
            """
        )
        conn.commit()

    def add(self, rows: Sequence[tuple]) -> None:
        self.conn.executemany(
            "INSERT INTO docs (doc_id, source, title, body, facets, date, ticket_key, thread_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    def optimize(self) -> None:
        """Merge the FTS b-trees into one, which roughly halves query time."""
        self.conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
        self.conn.commit()

    def count(self) -> int:
        """Number of indexed documents, cached in a side table.

        `count(*)` on an FTS5 table is a full scan of several gigabytes, so
        calling it once per query term - as the term filter naively did -
        costs more than the search it was meant to speed up.
        """
        if self._total is not None:
            return self._total
        row = self.conn.execute("SELECT df FROM term_df WHERE term = ?", ("\x00total",)).fetchone()
        if row is None:
            total = int(self.conn.execute("SELECT count(*) FROM docs").fetchone()[0])
            self.conn.execute("INSERT OR REPLACE INTO term_df VALUES (?, ?)", ("\x00total", total))
            self.conn.commit()
        else:
            total = int(row[0])
        self._total = total
        return total

    # --- read ----------------------------------------------------------------

    def document_frequency(self, term: str) -> int:
        """How many documents contain a term, memoised on disk.

        Counting costs a few milliseconds the first time a term is seen and
        nothing thereafter, which matters because questions reuse vocabulary.
        """
        row = self.conn.execute("SELECT df FROM term_df WHERE term = ?", (term,)).fetchone()
        if row is not None:
            return int(row[0])
        try:
            df = int(
                self.conn.execute(
                    "SELECT count(*) FROM docs WHERE docs MATCH ?", (f'"{term}"',)
                ).fetchone()[0]
            )
        except sqlite3.OperationalError:
            df = 0  # unparseable as a term; treat as matching nothing
        self.conn.execute("INSERT OR REPLACE INTO term_df VALUES (?, ?)", (term, df))
        self.conn.commit()
        return df

    def selective_terms(self, question: str, mute: Sequence[str] = ()) -> list[str]:
        """The terms in a question worth actually searching for.

        Anything matching more than `MAX_DF_FRACTION` of the corpus is
        discarded: it cannot discriminate between documents, and evaluating it
        dominates the query. On this corpus that removes `request` (346k docs)
        and `api` (182k) while keeping `multipart` (1,813) — which is the
        difference between a three-second search and a fast one.
        """
        # Mute before selecting, never after. The rarest-terms fallback runs
        # over whatever is left, so removing the person's name afterwards threw
        # away the good terms and kept the filler: "what did Jonas Weber say
        # about capacity" ended up searching for `say` and `did`.
        muted = {word for phrase in mute for word in query_terms(phrase)}
        ceiling = (self.count() or 1) * MAX_DF_FRACTION
        scored = [
            (t, self.document_frequency(t))
            for t in query_terms(question)
            if t not in muted
        ]
        present = [(t, df) for t, df in scored if df > 0]
        keep = [t for t, df in present if df <= ceiling]
        if len(keep) < MIN_TERMS:
            # Every term is common. Take the rarest few rather than nothing.
            keep = [t for t, _ in sorted(present, key=lambda x: x[1])[:MIN_TERMS]]
        return keep

    def topic_terms(self, question: str, mute: Sequence[str] = ()) -> list[str]:
        """Content terms for ranking an already-bounded candidate scope.

        Global document frequency is intentionally not a cutoff here. A common
        enterprise word such as `policy` is weak across 500k documents but very
        useful when ranking only a person's direct graph neighbors.
        """
        muted = {word for phrase in mute for word in query_terms(phrase)}
        return [
            term
            for term in query_terms(question)
            if term not in muted
            and term not in _QUESTION_TERM
            and self.document_frequency(term) > 0
        ]

    def search(
        self,
        question: str,
        limit: int = 20,
        source: str | None = None,
        also: Sequence[str] = (),
        drop: Sequence[str] = (),
    ) -> list[Candidate]:
        """The `limit` best-matching documents for a question.

        The title is weighted 4x the body: in this corpus the subject line
        often carries the answer the body only gestures at.

        `also` carries terms the caller has established are worth searching for
        even though the question did not contain them — in practice, the other
        names a person is known by. They bypass the frequency filter: a
        surname can be common in the corpus and still be the most important
        thing in the query, because it is the thing that identifies who is
        being asked about.
        """
        # Words belonging to the person being asked about are removed from the
        # topic half. Left in, they appear on both sides of the conjunction and
        # it collapses to "any document mentioning this person at all" — which
        # is how "what did Jonas Weber say about capacity" returned twenty
        # documents about onboarding and invoices.
        terms = self.selective_terms(question, mute=drop)
        extra = [t for t in dict.fromkeys(also) if t]
        if not terms and not extra:
            return []

        if terms and extra:
            # Conjunction, not union. OR-ing twenty of a person's email
            # addresses into the query buries the subject of the question
            # entirely — every document they ever appeared in outranks the
            # documents about the thing being asked. What is wanted is
            # narrower than either half: documents about this topic that
            # mention this person *under any of their names*.
            topic = " OR ".join(f'"{t}"' for t in terms)
            who = " OR ".join(f'"{t}"' for t in extra)
            match = f"({topic}) AND ({who})"
        else:
            match = " OR ".join(f'"{t}"' for t in (terms or extra))
        sql = (
            # title 4x, facets 2x, body 1x. The facet column is a handful of
            # short precise strings rather than prose, so a match in it is a
            # stronger signal per token than a body match and a weaker one than
            # a title match -- and BM25 already divides by field length, so the
            # weight is the only thumb on the scale.
            "SELECT doc_id, source, title, body, date,"
            " bm25(docs, 0.0, 0.0, 4.0, 1.0, 2.0) AS rank"
            " FROM docs WHERE docs MATCH ?"
        )
        params: list = [match]
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        return [
            Candidate(
                doc_id=r["doc_id"],
                source=r["source"],
                title=r["title"] or "",
                body=r["body"] or "",
                date=r["date"] or "",
                # BM25 returns negative numbers, better being more negative.
                # Flipped so callers can read it as "higher is better".
                score=-float(r["rank"]),
            )
            for r in self.conn.execute(sql, params)
        ]

    def match_count(self, question: str) -> int:
        """How many documents the question matches before ranking trims them.

        `search` returns a page; this is the size of the field it was chosen
        from. Reporting the page as though it were the search made the trace
        read as if twenty documents had been looked at, when the real number
        for an ordinary question is six figures. FTS5 keeps this count, so
        asking for it costs nothing.
        """
        terms = self.selective_terms(question)
        if not terms:
            return 0
        match = " OR ".join(f'"{t}"' for t in terms)
        return int(
            self.conn.execute(
                "SELECT count(*) FROM docs WHERE docs MATCH ?", (match,)
            ).fetchone()[0]
        )

    def build_docmap(self) -> int:
        """Map every `doc_id` to its rowid, so fetches stop scanning.

        `docs` is an FTS5 virtual table and `doc_id` is UNINDEXED, which means
        `WHERE doc_id IN (...)` has no index to use and SQLite reads all
        511,962 rows -- 21 seconds to fetch the 55 documents the graph had
        just selected, which is the whole cost of a graph-scoped question.
        FTS5 does answer rowid lookups directly, so one ordinary table from
        doc_id to rowid restores that. Rebuilding is idempotent and only
        needs doing after the index changes.
        """
        conn = self.conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS docmap (doc_id TEXT PRIMARY KEY, rid INTEGER)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO docmap (doc_id, rid) SELECT doc_id, rowid FROM docs"
        )
        conn.commit()
        return int(conn.execute("SELECT count(*) FROM docmap").fetchone()[0])

    def _rowids(self, doc_ids: Sequence[str]) -> dict[str, int]:
        """Rowids for the ids we know, silently skipping the ones we do not."""
        try:
            placeholders = ", ".join("?" for _ in doc_ids)
            rows = self.conn.execute(
                f"SELECT doc_id, rid FROM docmap WHERE doc_id IN ({placeholders})",
                list(doc_ids),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}  # map not built yet; callers fall back to a scan
        return {row["doc_id"]: int(row["rid"]) for row in rows}

    def _rows_by_rowid(self, rowids: Sequence[int]) -> dict[int, sqlite3.Row]:
        if not rowids:
            return {}
        placeholders = ", ".join("?" for _ in rowids)
        rows = self.conn.execute(
            "SELECT rowid AS rid, doc_id, source, title, body, date FROM docs "
            f"WHERE rowid IN ({placeholders})",
            list(rowids),
        ).fetchall()
        return {int(row["rid"]): row for row in rows}

    @staticmethod
    def _candidate(row: sqlite3.Row) -> Candidate:
        return Candidate(
            doc_id=row["doc_id"],
            source=row["source"],
            title=row["title"] or "",
            body=row["body"] or "",
            date=row["date"] or "",
            score=0.0,
        )

    def get(self, doc_id: str) -> Candidate | None:
        found = self.get_many([doc_id])
        return found[0] if found else None

    def get_many(self, doc_ids: Sequence[str]) -> list[Candidate]:
        """Fetch known document ids without changing the caller's requested order."""
        if not doc_ids:
            return []
        unique_ids = list(dict.fromkeys(doc_ids))
        rowids = self._rowids(unique_ids)
        if rowids:
            by_rowid = self._rows_by_rowid(list(rowids.values()))
            found = {
                doc_id: self._candidate(by_rowid[rid])
                for doc_id, rid in rowids.items()
                if rid in by_rowid
            }
        else:
            placeholders = ", ".join("?" for _ in unique_ids)
            rows = self.conn.execute(
                "SELECT doc_id, source, title, body, date FROM docs "
                f"WHERE doc_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
            found = {row["doc_id"]: self._candidate(row) for row in rows}
        return [found[doc_id] for doc_id in doc_ids if doc_id in found]

    def search_scoped(
        self, question: str, doc_ids: Sequence[str], limit: int = 20, drop: Sequence[str] = ()
    ) -> list[Candidate]:
        """Topic-rank a bounded, already-chosen set of documents.

        The graph hands over a few hundred documents at most, and ranking them
        is not a search problem. Two FTS-based attempts both cost about 25
        seconds on a real question: SQLite evaluates `docs MATCH` against the
        whole index before joining it to the scope, so restricting to 55
        documents restricts nothing -- the union of six ordinary English terms
        still matches a large fraction of half a million documents, and every
        one of them gets ranked.

        Fetching the scope by primary key and scoring it here is milliseconds,
        because the work is proportional to the scope rather than the corpus.
        This is only safe because the caller caps the scope; it is not a
        general search path and must not be used as one.

        Documents covering more distinct topic terms rank first, so a short
        document carrying one generic query verb cannot outrank one carrying
        the real multi-word topic. Rarer terms count for more, and the title
        counts for more than the body -- in this corpus the subject line often
        states what the body only gestures at.
        """
        ids = list(dict.fromkeys(doc_ids))
        if not ids or limit <= 0:
            return []
        terms = self.topic_terms(question, mute=drop)
        if not terms:
            return []
        terms = sorted(terms, key=self.document_frequency)[:MAX_SCOPE_TERMS]
        weights = {
            term: 1.0 / math.log(2 + self.document_frequency(term)) for term in terms
        }

        scored: list[tuple[int, float, int, Candidate]] = []
        for position, document in enumerate(self.get_many(ids)):
            title = (document.title or "").lower()
            body = (document.body or "").lower()
            coverage = 0
            weight = 0.0
            for term in terms:
                in_title = title.count(term)
                in_body = body.count(term)
                if in_title or in_body:
                    coverage += 1
                    weight += weights[term] * (4.0 * in_title + min(in_body, 8))
            if coverage:
                scored.append((-coverage, -weight, position, document))
        scored.sort(key=lambda item: item[:3])
        return [
            Candidate(
                d.doc_id, d.source, d.title, d.body, d.date, round(-weight, 4)
            )
            for _, weight, _, d in scored[:limit]
        ]


# The header fields worth searching. `from` and `to` carry the people a mail
# question is about; `subject` restates the topic in the words the asker is
# likely to reuse; `attachments` is the only place a filename appears at all.
_HEADER_FIELDS = ("from", "to", "cc", "subject", "attachments")

# What goes into the searchable facet column. Emails and mentions are
# deliberately absent: the ontology already resolves people and feeds them back
# through `search(also=...)`, and duplicating every address here would put a
# second copy of every name in the index for the identity path to trip over.
_FACET_FIELDS = ("slug", "folders", "channels", "speakers", "attendees")


def _facet_text(doc: dict) -> str:
    """The filing details, flattened into one searchable string.

    Hyphens and underscores are left in place rather than split by hand:
    `unicode61` already tokenises `customer-success-and-support` into its four
    words, so "the customer success and support space" matches the folder it
    names without the caller having to guess the punctuation.
    """
    parts: list[str] = []
    for field in _FACET_FIELDS:
        value = doc.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value if isinstance(item, (str, int)))
    headers = doc.get("headers")
    if isinstance(headers, dict):
        parts.extend(
            str(headers[field]) for field in _HEADER_FIELDS
            if isinstance(headers.get(field), str)
        )
    return " ".join(part for part in parts if part)


def iter_normalized(sources: Sequence[str]) -> Iterator[tuple]:
    """Stream normalized shards as index rows."""
    for source in sources:
        path = NORMALIZED / f"{source}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                yield (
                    doc.get("doc_id") or "",
                    doc.get("source") or source,
                    doc.get("title") or "",
                    doc.get("body") or "",
                    _facet_text(doc),
                    doc.get("date") or "",
                    doc.get("ticket_key") or "",
                    doc.get("thread_id") or "",
                )
