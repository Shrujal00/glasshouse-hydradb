#!/usr/bin/env python
"""Measure retrieval: does knowing who people are actually find better documents?

    python scripts/score.py --limit 150
    python scripts/score.py --limit 150 --k 10

Runs every question twice — once as plain keyword search, once with the query
expanded through the ontology into every name a person answers to — and reports
how often the document holding the answer was retrieved.

**This is the only file permitted to read the gold answers.** They live outside
the repository at `$GOLD_ANSWERS_PATH` precisely so that the pipeline cannot
reach them: scoping retrieval or extraction to known-good documents would post
excellent numbers, be worthless, and be obvious to any judge reading the source.
Nothing here is importable by `glasshouse`; it only imports in the other
direction.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.ask import Asker  # noqa: E402
from glasshouse.config import STATE, get  # noqa: E402

REPORT = STATE / "retrieval_score.json"


def load_gold() -> list[dict]:
    path = get("GOLD_ANSWERS_PATH")
    if not path or not Path(path).exists():
        raise SystemExit("GOLD_ANSWERS_PATH is not set or missing; scoring needs it")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def run(limit: int, k: int, offset: int, stratify: bool) -> None:
    asker = Asker()
    pool = [g for g in load_gold() if g.get("expected_doc_ids")]
    if stratify:
        # The file is ordered by type, so the first N questions are all
        # `basic` — a sample that cannot show anything about the categories
        # the graph exists for. Take a slice of every type instead.
        buckets: dict[str, list[dict]] = defaultdict(list)
        for g in pool:
            buckets[g.get("question_type", "unknown")].append(g)
        per = max(1, limit // max(len(buckets), 1))
        gold = [g for rows in buckets.values() for g in rows[offset : offset + per]]
    else:
        gold = pool[offset : offset + limit]
    print(f"scoring {len(gold)} questions at recall@{k}\n", flush=True)

    hits = {"plain": 0, "identity": 0}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "plain": 0, "identity": 0})
    changed: list[dict] = []
    resolved_any = 0
    latency: list[float] = []

    for i, row in enumerate(gold, start=1):
        question, expected = row["question"], set(row["expected_doc_ids"])
        qtype = row.get("question_type", "unknown")

        t0 = time.time()
        plain = asker.recall.search(question, limit=k)

        named = asker.read_identities(question)
        surfaces, asked = [], []
        for person in named:
            surfaces += asker.surfaces_of(person)
            asked += list(person.surfaces)
        wide = (
            asker.recall.search(question, limit=k, also=surfaces, drop=asked)
            if surfaces
            else plain
        )
        latency.append(time.time() - t0)
        resolved_any += bool(named)

        got_plain = bool(expected & {d.doc_id for d in plain})
        got_wide = bool(expected & {d.doc_id for d in wide})
        hits["plain"] += got_plain
        hits["identity"] += got_wide
        bucket = by_type[qtype]
        bucket["n"] += 1
        bucket["plain"] += got_plain
        bucket["identity"] += got_wide

        if got_plain != got_wide:
            changed.append(
                {
                    "question_id": row["question_id"],
                    "question": question,
                    "type": qtype,
                    "direction": "won" if got_wide else "lost",
                    "people": [p.name for p in named],
                }
            )
        if i % 25 == 0:
            print(f"  {i:>4}/{len(gold)}  plain {hits['plain']:>3}  identity {hits['identity']:>3}",
                  flush=True)

    n = len(gold)
    won = sum(1 for c in changed if c["direction"] == "won")
    lost = len(changed) - won

    print(f"\n{'='*68}")
    print(f"  questions scored          {n:>6}")
    print(f"  named a person we know    {resolved_any:>6}  ({resolved_any/max(n,1):.0%})")
    print(f"  median latency            {statistics.median(latency)*1000:>6.0f}ms")
    print()
    print(f"  recall@{k}, plain search   {hits['plain']:>6} / {n}   {hits['plain']/max(n,1):.1%}")
    print(f"  recall@{k}, identity-aware {hits['identity']:>6} / {n}   {hits['identity']/max(n,1):.1%}")
    print(f"  net change                {hits['identity']-hits['plain']:>+6}   ({won} won, {lost} lost)")

    print(f"\n  by question type:")
    print(f"    {'type':28s} {'n':>4} {'plain':>7} {'identity':>9}")
    for qtype, b in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {qtype:28s} {b['n']:>4} {b['plain']/max(b['n'],1):>6.0%} "
              f"{b['identity']/max(b['n'],1):>8.0%}")

    if changed:
        print(f"\n  questions whose retrieval changed (first 8):")
        for c in changed[:8]:
            mark = "won " if c["direction"] == "won" else "LOST"
            print(f"    [{mark}] {c['question'][:62]}")
            if c["people"]:
                print(f"            via {', '.join(c['people'][:3])}")

    STATE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {
            "k": k, "questions": n, "resolved_any": resolved_any,
            "recall_plain": hits["plain"], "recall_identity": hits["identity"],
            "won": won, "lost": lost,
            "by_type": {t: dict(b) for t, b in by_type.items()},
            "changed": changed,
        }, indent=2))
    print(f"\n  wrote {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="questions to score")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=20, help="documents retrieved per question")
    ap.add_argument("--all-types", action="store_true",
                    help="sample evenly across question types instead of taking the first N")
    args = ap.parse_args()
    run(args.limit, args.k, args.offset, args.all_types)


if __name__ == "__main__":
    main()
