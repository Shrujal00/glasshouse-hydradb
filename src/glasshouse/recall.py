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
import re
import sqlite3
from dataclasses import dataclass
from itertools import combinations
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
        self._conn: sqlite3.Connection | None = None
        self._total: int | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- build ---------------------------------------------------------------

    def create(self) -> None:
        """(Re)create the index. Metadata columns are unindexed - they are for
        filtering and citation, and indexing them would dilute BM25."""
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
            "INSERT INTO docs (doc_id, source, title, body, date, ticket_key, thread_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
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
            "SELECT doc_id, source, title, body, date, bm25(docs, 0.0, 0.0, 4.0, 1.0) AS rank"
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

    def get(self, doc_id: str) -> Candidate | None:
        row = self.conn.execute(
            "SELECT doc_id, source, title, body, date FROM docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return Candidate(
            doc_id=row["doc_id"],
            source=row["source"],
            title=row["title"] or "",
            body=row["body"] or "",
            date=row["date"] or "",
            score=0.0,
        )

    def get_many(self, doc_ids: Sequence[str]) -> list[Candidate]:
        """Fetch known document ids without changing the caller's requested order."""
        if not doc_ids:
            return []
        unique_ids = list(dict.fromkeys(doc_ids))
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self.conn.execute(
            "SELECT doc_id, source, title, body, date FROM docs WHERE doc_id IN (" + placeholders + ")",
            unique_ids,
        ).fetchall()
        found = {
            row["doc_id"]: Candidate(
                doc_id=row["doc_id"], source=row["source"], title=row["title"] or "",
                body=row["body"] or "", date=row["date"] or "", score=0.0,
            )
            for row in rows
        }
        return [found[doc_id] for doc_id in doc_ids if doc_id in found]

    def search_scoped(
        self, question: str, doc_ids: Sequence[str], limit: int = 20, drop: Sequence[str] = ()
    ) -> list[Candidate]:
        """Topic-rank a bounded graph-derived document scope with SQLite FTS5.

        Documents matching more distinct topic terms are selected first. This
        avoids a short document containing only a generic query verb outranking
        a document that contains the actual multi-word topic.
        """
        ids = list(dict.fromkeys(doc_ids))
        if not ids or limit <= 0:
            return []
        terms = self.topic_terms(question, mute=drop)
        if not terms:
            return []
        terms = sorted(terms, key=self.document_frequency)[:MAX_SCOPE_TERMS]
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS graph_scope (doc_id TEXT PRIMARY KEY)")
        self.conn.execute("DELETE FROM graph_scope")
        self.conn.executemany("INSERT INTO graph_scope (doc_id) VALUES (?)", ((doc_id,) for doc_id in ids))
        found: list[Candidate] = []
        seen: set[str] = set()
        for coverage in range(len(terms), 0, -1):
            clauses = [
                " AND ".join(f'"{term}"' for term in group)
                for group in combinations(terms, coverage)
            ]
            match = " OR ".join(f"({clause})" for clause in clauses)
            rows = self.conn.execute(
                "SELECT docs.doc_id, docs.source, docs.title, docs.body, docs.date, "
                "bm25(docs, 0.0, 0.0, 4.0, 1.0) AS rank "
                "FROM docs JOIN graph_scope ON docs.doc_id = graph_scope.doc_id "
                "WHERE docs MATCH ? ORDER BY rank LIMIT ?",
                # Higher-coverage documents reappear in every lower tier. Fetch
                # a full page so those duplicates do not consume the remaining
                # allowance before new fallback documents are seen.
                (match, limit),
            )
            for row in rows:
                if row["doc_id"] in seen:
                    continue
                seen.add(row["doc_id"])
                found.append(
                    Candidate(
                        row["doc_id"], row["source"], row["title"] or "",
                        row["body"] or "", row["date"] or "", -float(row["rank"]),
                    )
                )
                if len(found) >= limit:
                    break
            if len(found) >= limit:
                break
        return found


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
                    doc.get("date") or "",
                    doc.get("ticket_key") or "",
                    doc.get("thread_id") or "",
                )
