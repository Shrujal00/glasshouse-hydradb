#!/usr/bin/env python
"""Grade the answers themselves, not just the documents behind them.

    python scripts/grade.py --limit 44
    python scripts/grade.py --limit 44 --types conflicting_info,info_not_found

`scripts/score.py` measures whether the right document was retrieved. That is
upstream of the thing we are actually judged on: whether the answer is right.
The benchmark ships a rubric per question -- `answer_facts`, a list of natural
language statements the answer must support -- so grading means asking a model
to check each statement against what we wrote, one at a time, and counting.

Like `score.py` this reads gold data and nothing in `glasshouse` may import it.
The rubric never reaches the pipeline; it is applied only after the answer is
final, so the system cannot see what it is being marked against.
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

from glasshouse import answer as answer_module  # noqa: E402
from glasshouse.ask import Asker  # noqa: E402
from glasshouse.config import STATE, get  # noqa: E402

REPORT = STATE / "answer_grade.json"

JUDGE = """You are marking one answer against one required fact.

Reply with exactly one word: SUPPORTED or MISSING.

SUPPORTED means the answer states the required fact, or states something that
plainly entails it. Wording may differ. Extra detail is fine.
MISSING means the answer omits it, contradicts it, or hedges so much that a
reader would not come away knowing it.

If the required fact says the answer must decline or caveat, then SUPPORTED
means our answer actually declines or caveats."""


def load_gold() -> list[dict]:
    path = get("GOLD_ANSWERS_PATH")
    if not path or not Path(path).exists():
        raise SystemExit("GOLD_ANSWERS_PATH is not set or missing; grading needs it")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def judge(fact: str, written: str, model: str) -> bool:
    """One fact, one verdict. Kept separate so a long answer cannot blur them."""
    reply = answer_module._client().chat(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE},
            {
                "role": "user",
                "content": f"REQUIRED FACT:\n{fact}\n\nOUR ANSWER:\n{written}\n\nVerdict:",
            },
        ],
        options={"temperature": 0},
    )
    return "SUPPORTED" in reply["message"]["content"].upper()


def run(limit: int, types: list[str] | None, model: str) -> None:
    asker = Asker()
    pool = [g for g in load_gold() if g.get("answer_facts")]
    if types:
        pool = [g for g in pool if g.get("question_type") in types]

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        buckets[row.get("question_type", "unknown")].append(row)
    per = max(1, limit // max(len(buckets), 1))
    graded = [row for rows in buckets.values() for row in rows[:per]]

    print(f"grading {len(graded)} answers across {len(buckets)} types\n", flush=True)

    by_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "facts": 0, "supported": 0, "perfect": 0, "abstained": 0}
    )
    detail: list[dict] = []
    latency: list[float] = []

    for i, row in enumerate(graded, start=1):
        question = row["question"]
        qtype = row.get("question_type", "unknown")
        t0 = time.time()
        try:
            written = asker.ask(question).text
        except Exception as exc:
            written = ""
            print(f"  [{i}] pipeline error: {type(exc).__name__}: {exc}", flush=True)
        latency.append(time.time() - t0)

        facts = row["answer_facts"]
        supported = sum(judge(fact, written, model) for fact in facts) if written else 0
        bucket = by_type[qtype]
        bucket["n"] += 1
        bucket["facts"] += len(facts)
        bucket["supported"] += supported
        bucket["perfect"] += supported == len(facts)
        bucket["abstained"] += answer_module.NOT_FOUND in written or not written.strip()

        detail.append(
            {
                "question_id": row["question_id"],
                "type": qtype,
                "facts": len(facts),
                "supported": supported,
                "seconds": round(latency[-1], 1),
            }
        )
        print(
            f"  [{i:>3}/{len(graded)}] {qtype:24s} {supported}/{len(facts)} facts"
            f"  {latency[-1]:5.1f}s",
            flush=True,
        )

    total_facts = sum(b["facts"] for b in by_type.values())
    total_supported = sum(b["supported"] for b in by_type.values())
    total_perfect = sum(b["perfect"] for b in by_type.values())
    n = len(graded)

    print(f"\n{'='*72}")
    print(f"  answers graded            {n:>6}")
    print(f"  fact recall               {total_supported:>6} / {total_facts}"
          f"   {total_supported/max(total_facts,1):.1%}")
    print(f"  fully correct answers     {total_perfect:>6} / {n}"
          f"   {total_perfect/max(n,1):.1%}")
    print(f"  median latency            {statistics.median(latency):>6.1f}s")
    print(f"  p95 latency               "
          f"{sorted(latency)[int(len(latency)*0.95)-1]:>6.1f}s")
    print(f"\n  {'type':28s} {'n':>3} {'facts':>7} {'perfect':>8} {'abstained':>10}")
    for qtype, b in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {qtype:28s} {int(b['n']):>3} "
              f"{b['supported']/max(b['facts'],1):>6.0%} "
              f"{b['perfect']/max(b['n'],1):>7.0%} "
              f"{b['abstained']/max(b['n'],1):>9.0%}")

    STATE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {
            "model": model, "answers": n,
            "facts": total_facts, "supported": total_supported,
            "perfect": total_perfect,
            "median_seconds": round(statistics.median(latency), 1),
            "by_type": {t: dict(b) for t, b in by_type.items()},
            "detail": detail,
        }, indent=2))
    print(f"\n  wrote {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=33, help="answers to grade")
    ap.add_argument("--types", help="comma-separated question types")
    ap.add_argument("--model", default=None, help="judge model")
    args = ap.parse_args()
    run(
        args.limit,
        args.types.split(",") if args.types else None,
        args.model or answer_module.ADJUDICATION_MODEL,
    )


if __name__ == "__main__":
    main()
