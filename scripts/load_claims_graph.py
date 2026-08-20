#!/usr/bin/env python
"""Load the contradiction graph: what the company disagrees with itself about.

    python scripts/load_claims_graph.py --keys 250
    python scripts/load_claims_graph.py --keys 20 --dry-run
    python scripts/load_claims_graph.py --refresh          # re-extract, ignore checkpoint

`claims.py` already pulls competing assertions out of the evidence and
`trust.py` already decides between them -- but only for one question, in one
process, and the reasoning is thrown away the moment the answer is written. The
result is a system that can tell you two documents disagree if you happen to
ask about those two documents, and cannot tell you anything at all about the
shape of the disagreement across the company.

This puts the reasoning in HydraDB, where it stops being a step in a pipeline
and becomes something you can traverse:

    (:Claim {predicate, subject, object_value, asserted_at, trust, status})
    (:Claim)-[:EVIDENCED_BY]->(:Document)
    (:Claim)-[:ABOUT]->(:Entity)
    (:Claim)-[:CONTRADICTS]->(:Claim)
    (:Claim)-[:SUPERSEDES]->(:Claim)
    (:Disagreement)-[:OVER]->(:Claim)
    (:Disagreement)-[:CONCERNS]->(:Entity)

Three questions become one traversal each, and none of them is a question the
retrieval half of the system could answer at any price:

  * *What does this organisation contradict itself about?* -- rank the
    `Disagreement` nodes. Nobody has to have asked first.
  * *What was this value before, and what corrected it?* -- walk `SUPERSEDES`
    outwards from the current claim; every hop names the document that changed
    it.
  * *Who has been reading the version that turned out to be wrong?* -- anchor
    on the superseded claim, hop to the document that asserts it, hop back out
    to everyone the ontology connects to that document.

Which documents get read
------------------------
Not all 511,962, and not the ones the benchmark asks about. Claim extraction is
a model call, so the corpus has to be narrowed before it is read, and the
narrowing must not be a way of peeking at the questions -- nothing here touches
`$GOLD_ANSWERS_PATH` or `questions_blind.jsonl`.

The corpus narrows itself. Every document carries a `ticket_key` when it is one,
and 34,460 distinct work items are named across the nine tools -- but a key is
also *quoted*: `ENG-4821` appears in 249 GitHub documents, 236 Linear ones, 101
in Drive, 81 in Slack, 11 in Confluence. That quotation is the corpus saying
"these documents are about the same thing", which is exactly the precondition
for two of them to contradict each other, and it costs one FTS query per key to
measure. So the work items are ranked by how many *different tools* talk about
them, and the top `--keys` of them are read.

That ranking is also why the disagreements found here are worth showing. A
contradiction inside one Jira ticket is a typo. A contradiction between the
Confluence page, the Linear ticket and the Slack thread about one work item is
an organisation that has lost track of its own decision.

Claims are only allowed to contradict each other inside one work item -- see
`scope` in `claims.py`. Two unrelated tickets that both have an `owner` are two
facts, and grouping them by subject alone would manufacture disagreements out
of a naming coincidence.

Cost and restartability
-----------------------
Extraction is the expensive half and the only half that can fail: Ollama Cloud
session limits have been hit twice on this account, and a rate-limited run
records every answer as nothing found rather than as an error. So every claim
is appended to `data/state/claims_graph.jsonl` as it is extracted, a re-run
skips work items already in that file, and the write half can be replayed
against the graph as many times as necessary -- node and edge ids are digests
of stable keys, so `MERGE` overwrites in place.

Nothing is deleted first. `DETACH DELETE` over this graph returns
`429 resource_exhausted` from admission control, because deleting a vertex
scans its edges and `Document` has a million of them. Idempotent writes are not
a nicety here, they are the only way to load the graph twice.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse import claims as extraction  # noqa: E402
from glasshouse import trust  # noqa: E402
from glasshouse.config import STATE  # noqa: E402
from glasshouse.graph import GraphEngine, GraphError, node_id  # noqa: E402
from glasshouse.recall import LocalRecall  # noqa: E402

CHECKPOINT = STATE / "claims_graph.jsonl"
REPORT = STATE / "claims_graph.json"

BATCH = 500
WRITE_RETRIES = 3

# How many documents one work item is allowed to spend, and how many of them
# may come from a single tool. The cap on one tool is the point: eight Slack
# messages from one thread agree with each other by construction, and a work
# item that spends its whole budget inside one tool cannot produce the
# cross-tool disagreement this is looking for.
DOCS_PER_KEY = 9
DOCS_PER_SOURCE = 3

# Predicates whose *value* is a person rather than a thing. `owner of ENG-4824
# = "liam"` names Liam in the object, not the subject -- so resolving only
# subjects against the ontology left the two halves of the product unjoined:
# 57 ABOUT edges over 875 claims, and not one disagreement that named anybody.
PERSON_VALUED = ("owner", "reports_to")

# Reviewed artefacts first, so that when a work item has more documents than it
# is allowed the ones that survive are the ones a person would check.
SOURCE_ORDER = (
    "confluence",
    "jira",
    "linear",
    "github",
    "google_drive",
    "hubspot",
    "gmail",
    "fireflies",
    "slack",
)


# --- choosing what to read --------------------------------------------------


# FTS tokenizes `ENG-4821` into `eng 4821`, so a phrase match on a work item
# key also matches prose that merely happens to run those two tokens together
# -- `FIRST-100` matched 322 documents that way, most of them saying "first 100
# customers". The `LIKE` re-checks the literal key against the rows FTS already
# narrowed to, which costs nothing and is the difference between a work item
# and a coincidence. It runs second on purpose: on its own it is a scan of half
# a million bodies.
QUOTING = (
    "docs MATCH ?1 AND (ticket_key = ?2 OR title LIKE ?3 OR body LIKE ?3)"
)


def _quoting(key: str) -> tuple[str, str, str]:
    return (f'"{key}"', key, f"%{key}%")


def work_items(recall: LocalRecall, wanted: int, min_sources: int) -> list[dict]:
    """The work items the most different tools talk about.

    One FTS query per distinct `ticket_key` -- 34,460 of them in about twelve
    seconds, because a quoted key is a single term and the index is built for
    exactly this. Ranked by how many tools quote it, then by how much they say.
    """
    conn = recall.conn
    keys = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT ticket_key FROM docs WHERE ticket_key != ''"
        )
    ]
    ranked: list[dict] = []
    for key in keys:
        rows = conn.execute(
            f"SELECT source, count(*) AS n FROM docs WHERE {QUOTING} GROUP BY source",
            _quoting(key),
        ).fetchall()
        sources = {row["source"]: row["n"] for row in rows}
        if len(sources) < min_sources:
            continue
        ranked.append(
            {"key": key, "sources": sources, "documents": sum(sources.values())}
        )
    # Deterministic: tools first, then volume, then the key itself, so two runs
    # of the same corpus read the same work items in the same order.
    ranked.sort(key=lambda r: (-len(r["sources"]), -r["documents"], r["key"]))
    return ranked[:wanted]


def documents_for(recall: LocalRecall, key: str) -> list:
    """The documents one work item gets to spend, spread across its tools.

    A round over the sources in authority order taking one document each, until
    the budget runs out. That interleaving is what makes the selection
    cross-tool by construction rather than by luck -- taking the top nine by
    any single score fills the whole budget with Slack, which is 56% of the
    corpus.
    """
    rows = recall.conn.execute(
        f"SELECT doc_id, source, date FROM docs WHERE {QUOTING} LIMIT 400",
        _quoting(key),
    ).fetchall()
    by_source: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row["doc_id"])
    # A dated document beats an undated one: recency is half of how a conflict
    # is arbitrated, and an undated claim can never supersede anything.
    dated = {row["doc_id"] for row in rows if (row["date"] or "").strip()}
    for source, ids in by_source.items():
        ids.sort(key=lambda d: (d not in dated, d))

    picked: list[str] = []
    taken: dict[str, int] = defaultdict(int)
    while len(picked) < DOCS_PER_KEY:
        progressed = False
        for source in SOURCE_ORDER:
            ids = by_source.get(source) or []
            if taken[source] >= min(DOCS_PER_SOURCE, len(ids)):
                continue
            picked.append(ids[taken[source]])
            taken[source] += 1
            progressed = True
            if len(picked) >= DOCS_PER_KEY:
                break
        if not progressed:
            break
    return recall.get_many(picked)


# --- extraction -------------------------------------------------------------


def extract_key(recall: LocalRecall, key: str) -> list[extraction.Claim]:
    """Every claim the documents about one work item assert.

    Batched at the extractor's own ceiling. The work item is passed as both the
    question and the scope: as the question it decides which passages of a long
    document are shown, and as the scope it tells the extractor to write the
    same subject string for the same thing so two documents can be seen to
    disagree.
    """
    docs = documents_for(recall, key)
    found: list[extraction.Claim] = []
    for start in range(0, len(docs), extraction.MAX_CLAIM_DOCS):
        chunk = docs[start : start + extraction.MAX_CLAIM_DOCS]
        found.extend(extraction.extract(chunk, key, scope=key))
    return found


def extract_all(
    recall: LocalRecall, items: list[dict], workers: int, refresh: bool
) -> list[extraction.Claim]:
    """Extract every work item, appending as we go.

    One `LocalRecall` serves the whole pool: it already opens its SQLite
    connection per thread, because a connection belongs to the thread that
    opened it.

    Concurrency is modest on purpose. The engine is not the constraint here;
    Ollama Cloud is, and a burst that trips its session limit comes back as
    valid JSON asserting nothing, which would be written to the checkpoint as
    "this work item has no claims" and never retried.
    """
    done: set[str] = set()
    already: list[extraction.Claim] = []
    if CHECKPOINT.exists() and not refresh:
        torn = 0
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            # A run killed mid-write leaves a partial final line, and that is
            # precisely the run this checkpoint exists to recover -- a rate
            # limit or a Ctrl-C. Refusing to parse the whole file because of
            # its last few bytes would throw away every work item paid for
            # before the interruption.
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            if row.get("_scanned"):
                done.add(row["_scanned"])
                continue
            try:
                already.append(extraction.Claim(**row))
            except TypeError:
                # Written by an older extractor with different fields. Drop it
                # and let the work item be read again rather than load a claim
                # whose shape no longer matches what arbitration expects.
                torn += 1
                done.discard(row.get("scope", ""))
        print(f"  checkpoint holds {len(already):,} claims over {len(done):,} work items"
              + (f" ({torn} unreadable rows skipped)" if torn else ""))
    todo = [item for item in items if item["key"] not in done]
    if not todo:
        return already

    lock = threading.Lock()
    # `--refresh` truncates rather than appending. The checkpoint is keyed by
    # work item, not by extractor version, so leaving the old rows in place
    # would have the next ordinary run read back claims from a prompt that no
    # longer exists and silently mix two extractors in one graph.
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    handle = CHECKPOINT.open("w" if refresh else "a", encoding="utf-8")
    counts = {"keys": 0, "claims": 0, "empty": 0}
    fresh: list[extraction.Claim] = []
    def one(item: dict) -> None:
        try:
            claims = extract_key(recall, item["key"])
        except Exception as exc:  # a failed work item must not end the run
            with lock:
                print(f"  ! {item['key']}: {type(exc).__name__} {exc}"[:160], flush=True)
            return
        with lock:
            for claim in claims:
                handle.write(json.dumps(asdict(claim)) + "\n")
            # The scanned marker is what makes "no claims" distinguishable from
            # "not read yet". Without it every barren work item is re-extracted
            # on every run, at full price.
            handle.write(json.dumps({"_scanned": item["key"]}) + "\n")
            handle.flush()
            fresh.extend(claims)
            counts["keys"] += 1
            counts["claims"] += len(claims)
            counts["empty"] += not claims
            if counts["keys"] % 10 == 0 or counts["keys"] == len(todo):
                print(
                    f"  extracted {counts['keys']:>5,}/{len(todo):,} work items, "
                    f"{counts['claims']:>6,} claims, {counts['empty']:,} silent",
                    flush=True,
                )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, todo))
    finally:
        handle.close()
    # A run that finds nothing at all is a rate limit, not a quiet corpus. Say
    # so rather than writing an empty graph over a good one.
    if todo and not fresh:
        print(
            "  ! every work item came back empty — check for 429s before "
            "believing the corpus has no claims",
            flush=True,
        )
    return already + fresh


# --- turning arbitration into a graph ---------------------------------------


def _disagreement_key(scope: str, subject: str, predicate: str) -> str:
    seed = f"{scope}|{trust.subject_key(subject)}|{predicate}"
    return node_id(seed).__format__("x")[:16]


def _days_between(a: str, b: str) -> int:
    """Whole days between two `YYYY-MM-DD` strings, or 0 when either is absent.

    Approximate, and deliberately so: it is shown as "corrected 41 days later",
    where the reader wants the scale rather than the calendar.
    """
    try:
        first = [int(part) for part in a.split("-")]
        second = [int(part) for part in b.split("-")]
    except (ValueError, AttributeError):
        return 0
    if len(first) != 3 or len(second) != 3:
        return 0
    return abs(
        (first[0] - second[0]) * 365 + (first[1] - second[1]) * 30 + (first[2] - second[2])
    )


def resolve_subjects(engine: GraphEngine, subjects: list[str]) -> dict[str, tuple[str, str]]:
    """Which subjects name somebody the ontology already knows.

    One anchored hop each at about 0.09s, and there is no batched form -- the
    engine rejects the same statement under `UNWIND`. So this is bounded by the
    number of *distinct* subjects rather than by the number of claims, and an
    ambiguous form is dropped rather than guessed: a subject that reaches two
    people has not named either of them, which is the same rule identity
    resolution already applies at query time.
    """
    found: dict[str, tuple[str, str]] = {}
    for i, subject in enumerate(subjects, start=1):
        try:
            matches = engine.denoted_by(subject, limit=3)
        except Exception:
            continue
        if len(matches) == 1 and matches[0].entities <= 1:
            found[subject] = (matches[0].eid, matches[0].name)
        if i % 200 == 0:
            print(f"  resolved {i:,}/{len(subjects):,} subjects, {len(found):,} named someone",
                  flush=True)
    return found


def build(
    arbitration: trust.Arbitration,
    titles: dict[str, str],
    people: dict[str, tuple[str, str]],
    gen: str = "",
) -> dict[str, list[dict]]:
    """Every node and edge the contradiction graph needs, as writable rows."""
    rows: dict[str, list[dict]] = defaultdict(list)
    seen_claims: set[str] = set()

    def claim_row(claim: extraction.Claim) -> None:
        if claim.claim_id in seen_claims:
            return
        seen_claims.add(claim.claim_id)
        rows["claims"].append(
            {
                "id": node_id(f"claim:{claim.claim_id}"),
                "claim_id": claim.claim_id,
                "scope": claim.scope,
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object_value": claim.object_value,
                "doc_id": claim.doc_id,
                "title": titles.get(claim.doc_id, "")[:200],
                "source": claim.source,
                "asserted_at": claim.asserted_at,
                "extractor_confidence": float(claim.extractor_confidence),
                "trust": float(claim.trust),
                "status": claim.status,
                "rationale": claim.rationale[:500],
                "gen": gen,
            }
        )
        rows["evidenced_by"].append(
            {
                "src": node_id(f"claim:{claim.claim_id}"),
                "dst": node_id(f"doc:{claim.doc_id}"),
            }
        )
        # A claim reaches a person through whichever end names one. `owner`
        # and `reports_to` name them in the value; everything else, if at all,
        # in the subject.
        for who in (
            people.get(claim.object_value) if claim.predicate in PERSON_VALUED else None,
            people.get(claim.subject),
        ):
            if not who:
                continue
            rows["about"].append(
                {
                    "src": node_id(f"claim:{claim.claim_id}"),
                    "dst": node_id(f"entity:{who[0]}"),
                    "predicate": claim.predicate,
                }
            )
            break

    for claim in arbitration.claims:
        claim_row(claim)

    for conflict in arbitration.conflicts:
        members = [conflict.winner, *conflict.losers]
        # Two documents disagreeing needs two documents. Competing values that
        # all come out of one transcript are one document being read twice --
        # a meeting that mentions three different thresholds is three facts,
        # and extraction filing them under one subject is the failure mode
        # this guards, not a disagreement the organisation actually has. The
        # claims are still written and still queryable; what they do not get
        # is a node on a map that says the company contradicts itself.
        if len({claim.doc_id for claim in members}) < 2:
            continue
        key = _disagreement_key(conflict.scope, conflict.subject, conflict.predicate)
        anchor = node_id(f"disagreement:{key}")

        # Grouped exactly the way arbitration grouped, fold included. This
        # used to re-bucket on `_value` alone, which quietly disagreed with
        # the decision it was recording: `liam` and `liam + maria` came out as
        # two opposing sides with a CONTRADICTS edge between them, while
        # arbitration had already folded them into one position. A second
        # implementation of a grouping rule is a second chance to drift from
        # it, so this calls the same one.
        raw: dict[str, list[extraction.Claim]] = defaultdict(list)
        for claim in members:
            raw[trust._value(claim)].append(claim)
        by_value = trust._fold(raw)
        sources = sorted({claim.source for claim in members if claim.source})
        dates = sorted(d for d in (claim.asserted_at for claim in members) if d)
        runner = conflict.losers[0]

        # Every claim on one side contradicts every claim on every other side,
        # in both directions. Undirected would be the honest shape and the
        # engine has no undirected edge, so the pair is written twice: a
        # traversal that could only be walked from the winning side would make
        # the losing claim a dead end, and the losing claim is the one whose
        # blast radius somebody wants.
        values = list(by_value.values())
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                for a in left:
                    for b in right:
                        gap = round(abs(a.trust - b.trust), 4)
                        for src, dst in ((a, b), (b, a)):
                            rows["contradicts"].append(
                                {
                                    "src": node_id(f"claim:{src.claim_id}"),
                                    "dst": node_id(f"claim:{dst.claim_id}"),
                                    "gap": gap,
                                    "predicate": conflict.predicate,
                                    "scope": conflict.scope,
                                    "decided": int(conflict.decided),
                                }
                            )

        # `SUPERSEDES` only where recency is what settled it. `trust.py` marks
        # the loser `superseded` exactly then, and `disputed` when the two
        # values are simply in conflict -- an unresolved disagreement is not a
        # history of a fact and drawing it as one would be a lie about time.
        if conflict.decided:
            for loser in conflict.losers:
                if loser.status != "superseded":
                    continue
                rows["supersedes"].append(
                    {
                        "src": node_id(f"claim:{conflict.winner.claim_id}"),
                        "dst": node_id(f"claim:{loser.claim_id}"),
                        "days": _days_between(conflict.winner.asserted_at, loser.asserted_at),
                        "predicate": conflict.predicate,
                    }
                )

        named = (
            (people.get(conflict.winner.object_value)
             if conflict.predicate in PERSON_VALUED else None)
            or (people.get(runner.object_value)
                if conflict.predicate in PERSON_VALUED else None)
            or people.get(conflict.winner.subject)
            or people.get(runner.subject)
        )
        contested = min(conflict.winner.trust, runner.trust)
        rows["disagreements"].append(
            {
                "id": anchor,
                "key": key,
                "scope": conflict.scope,
                "subject": conflict.winner.subject,
                "predicate": conflict.predicate,
                "sides": len(by_value),
                "claims": len(members),
                "documents": len({claim.doc_id for claim in members}),
                "sources": "|".join(sources),
                "trust_gap": round(conflict.winner.trust - runner.trust, 4),
                "decided": int(conflict.decided),
                "winner_value": conflict.winner.object_value,
                "winner_source": conflict.winner.source,
                "winner_trust": float(conflict.winner.trust),
                "winner_claim_id": conflict.winner.claim_id,
                "runner_value": runner.object_value,
                "runner_source": runner.source,
                "runner_trust": float(runner.trust),
                "runner_claim_id": runner.claim_id,
                "rationale": conflict.rationale[:500],
                "first_asserted": dates[0] if dates else "",
                "last_asserted": dates[-1] if dates else "",
                "entity_eid": named[0] if named else "",
                "entity_name": named[1] if named else "",
                # What to look at first: how much is being said, across how
                # many tools, weighted up when both sides are credible and
                # doubled when arbitration refused to choose. A disagreement
                # nobody can settle between two trusted sources is the one a
                # person needs to see; a Slack aside losing to a Confluence
                # page is not.
                "weight": round(
                    len(members)
                    * (1 + len(sources))
                    * (1.0 + contested)
                    * (2.0 if not conflict.decided else 1.0),
                    3,
                ),
                "gen": gen,
            }
        )
        for side, (_value, holders) in enumerate(by_value.items()):
            for claim in holders:
                rows["over"].append(
                    {
                        "src": anchor,
                        "dst": node_id(f"claim:{claim.claim_id}"),
                        "side": side,
                        "status": claim.status,
                    }
                )
        if named:
            rows["concerns"].append({"src": anchor, "dst": node_id(f"entity:{named[0]}")})
    return rows


# --- writing ----------------------------------------------------------------


def write(engine: GraphEngine, rows: dict[str, list[dict]]) -> None:
    def flush(fn, items, what):
        items = _dedupe(items)
        for start in range(0, len(items), BATCH):
            chunk = items[start : start + BATCH]
            for attempt in range(WRITE_RETRIES):
                try:
                    fn(chunk)
                    break
                except GraphError:
                    if attempt == WRITE_RETRIES - 1:
                        raise
                    time.sleep(0.4 * (attempt + 1))
        print(f"  {what:<14} {len(items):>7,}", flush=True)

    # Nodes before edges, always: `merge_edges` matches existing endpoints and
    # writes nothing at all for an endpoint that is not there yet, silently.
    flush(
        lambda c: engine.upsert_nodes(
            "Claim",
            c,
            [
                "claim_id", "scope", "subject", "predicate", "object_value",
                "doc_id", "title", "source", "asserted_at",
                "extractor_confidence", "trust", "status", "rationale", "gen",
            ],
        ),
        rows["claims"],
        "claims",
    )
    flush(
        lambda c: engine.upsert_nodes(
            "Disagreement",
            c,
            [
                "key", "scope", "subject", "predicate", "sides", "claims",
                "documents", "sources", "trust_gap", "decided",
                "winner_value", "winner_source", "winner_trust", "winner_claim_id",
                "runner_value", "runner_source", "runner_trust", "runner_claim_id",
                "rationale", "first_asserted", "last_asserted",
                "entity_eid", "entity_name", "weight", "gen",
            ],
        ),
        rows["disagreements"],
        "disagreements",
    )
    flush(
        lambda c: engine.merge_edges("EVIDENCED_BY", c, [], "Claim", "Document"),
        rows["evidenced_by"],
        "evidenced_by",
    )
    flush(
        lambda c: engine.merge_edges("ABOUT", c, ["predicate"], "Claim", "Entity"),
        rows["about"],
        "about",
    )
    flush(
        lambda c: engine.merge_edges(
            "CONTRADICTS", c, ["gap", "predicate", "scope", "decided"], "Claim"
        ),
        rows["contradicts"],
        "contradicts",
    )
    flush(
        lambda c: engine.merge_edges("SUPERSEDES", c, ["days", "predicate"], "Claim"),
        rows["supersedes"],
        "supersedes",
    )
    flush(
        lambda c: engine.merge_edges("OVER", c, ["side", "status"], "Disagreement", "Claim"),
        rows["over"],
        "over",
    )
    flush(
        lambda c: engine.merge_edges("CONCERNS", c, [], "Disagreement", "Entity"),
        rows["concerns"],
        "concerns",
    )


def _dedupe(items: list[dict]) -> list[dict]:
    """One row per (src, dst) or per id. Two claims can reach the same document
    and the same batch would then carry the same edge twice, which the engine
    accepts and which makes every count downstream wrong."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for item in items:
        key = (item.get("id"), item.get("src"), item.get("dst"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# --- run --------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    recall = LocalRecall()
    if not recall.path.exists():
        raise SystemExit("no recall index; run scripts/build_index.py first")

    t0 = time.time()
    print(f"ranking work items by how many tools quote them ...", flush=True)
    items = work_items(recall, args.keys, args.min_sources)
    if not items:
        raise SystemExit("no work item is quoted by enough tools; lower --min-sources")
    spread = sum(len(i["sources"]) for i in items) / len(items)
    print(
        f"  {len(items):,} work items, {spread:.1f} tools each on average, "
        f"widest is {items[0]['key']} across {len(items[0]['sources'])}",
        flush=True,
    )
    if args.dry_run:
        for item in items[:20]:
            tools = ", ".join(f"{s}×{n}" for s, n in sorted(item["sources"].items()))
            print(f"  {item['key']:<16} {item['documents']:>5,} docs   {tools}")
        return

    print(f"extracting claims ({args.workers} at a time) ...", flush=True)
    found = extract_all(recall, items, args.workers, args.refresh)
    if not found:
        raise SystemExit("no claims extracted; nothing to load")
    print(f"  {len(found):,} claims from {len({c.scope for c in found}):,} work items")

    print("arbitrating ...", flush=True)
    arbitration = trust.arbitrate(found)
    undecided = sum(1 for c in arbitration.conflicts if not c.decided)
    single = sum(
        1
        for c in arbitration.conflicts
        if len({m.doc_id for m in (c.winner, *c.losers)}) < 2
    )
    print(
        f"  {len(arbitration.claims):,} settled claims, "
        f"{len(arbitration.conflicts):,} disagreements, "
        f"{undecided:,} of them the system refuses to settle",
        flush=True,
    )
    if single:
        print(
            f"  {single:,} dropped as single-document — one text read twice is "
            f"not two sources disagreeing",
            flush=True,
        )

    engine = GraphEngine()
    if not engine.wait_until_ready(90):
        raise SystemExit("engine not reachable; is `docker compose up -d` running?")

    print("resolving subjects against the identity graph ...", flush=True)
    subjects = sorted(
        {claim.subject for claim in arbitration.claims}
        | {c.object_value for c in arbitration.claims if c.predicate in PERSON_VALUED}
    )
    people = resolve_subjects(engine, subjects) if not args.no_resolve else {}
    print(f"  {len(people):,}/{len(subjects):,} subjects and owner-values name "
          f"somebody the ontology knows")

    titles = {
        doc.doc_id: doc.title
        for doc in recall.get_many(sorted({c.doc_id for c in arbitration.claims}))
    }
    # Nothing in this graph can be deleted -- admission control refuses
    # `DETACH DELETE` even for one anchored node -- so a reload cannot replace
    # the previous one, only sit alongside it. Every load stamps its nodes and
    # records the stamp here; the reader asks for one stamp and sees one map.
    gen = f"{int(t0)}"
    rows = build(arbitration, titles, people, gen)
    print("writing ...", flush=True)
    write(engine, rows)

    stats = engine.claim_stats()
    elapsed = time.time() - t0
    print(
        f"\nloaded in {elapsed:.1f}s — graph now holds {stats['claims']:,} claims, "
        f"{stats['disagreements']:,} disagreements, "
        f"{stats['contradicts']:,} CONTRADICTS, {stats['supersedes']:,} SUPERSEDES"
    )
    REPORT.write_text(
        json.dumps(
            {
                "gen": gen,
                "work_items": len(items),
                "claims_extracted": len(found),
                "claims_settled": len(arbitration.claims),
                "disagreements": len(arbitration.conflicts),
                "undecided": undecided,
                "subjects_resolved": len(people),
                "graph": stats,
                "seconds": round(elapsed, 1),
            },
            indent=2,
        )
        + "\n"
    )

    worst = sorted(rows["disagreements"], key=lambda r: -r["weight"])[:6]
    for row in worst:
        verdict = "settled" if row["decided"] else "REFUSES TO DECIDE"
        print(
            f"  {row['scope']:<14} {row['predicate']:<9} of {row['subject'][:34]:<34} "
            f"{row['sides']} values across {row['sources'].replace('|', ', ')}  [{verdict}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=int, default=250, help="work items to read")
    ap.add_argument(
        "--min-sources", type=int, default=3, help="tools that must quote a work item"
    )
    ap.add_argument("--workers", type=int, default=4, help="concurrent extractions")
    ap.add_argument("--refresh", action="store_true", help="ignore the claim checkpoint")
    ap.add_argument("--no-resolve", action="store_true", help="skip ABOUT edges")
    ap.add_argument("--dry-run", action="store_true", help="show the work items and stop")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
