#!/usr/bin/env python
"""The questions keyword search cannot express.

    python scripts/graph_queries.py

`scripts/ablate.py` asks whether the graph retrieves *documents* better than
BM25. This asks a different question: what happens to the four things the graph
was built for when you take the graph away?

Keyword search is given the same question text, so the comparison is fair. It
returns documents, because that is all it can return. Returning documents is
not an answer to "what does this company contradict itself about" -- nobody
typed a question, and there is no word to search for.

Reads no gold data. Everything here is the running system.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.ask import Asker  # noqa: E402
from glasshouse.config import STATE  # noqa: E402

REPORT = STATE / "graph_queries.json"


def timed(fn):
    started = time.perf_counter()
    try:
        return fn(), (time.perf_counter() - started) * 1000, None
    except Exception as exc:
        return None, (time.perf_counter() - started) * 1000, f"{type(exc).__name__}"


def main() -> None:
    asker = Asker()
    engine = asker.engine
    rows = []

    print("\n  four questions, asked of the graph and of keyword search\n")

    # 1 — identity. One anchored hop from a written form to a person.
    who, ms, err = timed(lambda: engine.denoted_by("elliot price", limit=8))
    rows.append(
        {
            "question": "Who is `elliot price`?",
            "graph": f"{len(who or [])} match, {who[0].alias_count if who else 0} written forms",
            "ms": ms,
            "keyword": "returns documents mentioning the words, not the person",
            "error": err,
        }
    )

    # 2 — containers. The hop that narrows a corpus to a folder.
    from glasshouse.graph import node_id

    key = "confluence:folder:customer-success-and-support"
    inside, ms, err = timed(
        lambda: engine.documents_in_containers([(key, node_id(f"container:{key}"))], 400)
    )
    rows.append(
        {
            "question": "Which pages are in the customer-success space?",
            "graph": f"{len(inside or [])} documents, scoped from 511,962",
            "ms": ms,
            "keyword": "cannot scope — the folder name is not in the page bodies",
            "error": err,
        }
    )

    # 3 — the one with no keyword equivalent at all.
    dis, ms, err = timed(lambda: engine.disagreements(limit=200))
    undecided, ms2, err2 = timed(lambda: engine.disagreements(limit=200, undecided_only=True))
    rows.append(
        {
            "question": "What does this company contradict itself about?",
            "graph": f"{len(dis or [])} disagreements, {len(undecided or [])} it refuses to settle",
            "ms": ms,
            "keyword": "no query exists — nobody typed a question",
            "error": err or err2,
        }
    )

    # 4 — the multi-hop one. Claim to the document that asserts it, then back
    # out to everyone the ontology attached to that document.
    #
    # Not every claim reaches people: the hop only fires when the ontology
    # resolved somebody in the document behind the claim, and most claim-bearing
    # documents are tickets where it did not. Scan for one that does, and report
    # the coverage rather than the first row.
    def widest():
        best, scanned, reached = [], 0, 0
        for d in (dis or [])[:40]:
            if not d.winner_claim_id:
                continue
            scanned += 1
            people = engine.blast_radius(d.winner_claim_id, limit=60)
            if people:
                reached += 1
                if len(people) > len(best):
                    best = people
        return best, scanned, reached

    (blast, scanned, reached), _scan_ms, err = timed(widest)
    # Time one hop, not the scan that located it.
    _, ms, _ = timed(
        lambda: engine.blast_radius(
            next(d.winner_claim_id for d in dis if engine.blast_radius(d.winner_claim_id, 60)),
            limit=60,
        )
        if blast
        else []
    )
    rows.append(
        {
            "question": "Who read the version that turned out to be wrong?",
            "graph": f"{len(blast or [])} people, claim → document → person",
            "coverage": f"{reached} of {scanned} claims reach people this way",
            "scan_ms": _scan_ms,
            "ms": ms,
            "keyword": "cannot traverse — there is no edge to follow",
            "error": err,
        }
    )

    # What BM25 does when handed the same four questions.
    print(f"  {'question':<52}{'HydraDB':>26}{'ms':>9}")
    print("  " + "-" * 88)
    for r in rows:
        note = r["graph"] if not r["error"] else f"FAILED — {r['error']}"
        print(f"  {r['question']:<52}{note:>26}{r['ms']:>9.0f}")

    print(f"\n  {'the same question, given to keyword search':<52}{'result':>26}")
    print("  " + "-" * 88)
    for r in rows:
        hits = asker.recall.search(r["question"], limit=20)
        print(f"  {r['question']:<52}{f'{len(hits)} documents':>26}")
        r["keyword_documents"] = len(hits)

    print(
        "\n  Keyword search answers all four with a pile of documents, which is\n"
        "  not an answer to any of them. Turn the engine off and the left column\n"
        "  is empty — there is nothing else in the stack that can produce it.\n"
    )

    REPORT.write_text(json.dumps(rows, indent=2))
    print(f"  wrote {REPORT}\n")


if __name__ == "__main__":
    main()
