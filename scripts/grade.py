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
    """One fact, one verdict. Kept separate so a long answer cannot blur them.

    Retried, because a full run is hours long and the connection to the model
    is the least reliable thing in it. An unretried blip in hour two used to
    raise straight out of the loop and end the run -- which was survivable only
    because the report is checkpointed, and is still an hour of grading that
    has to be resumed by hand.

    A fact that cannot be judged after the retries counts as unsupported.
    Guessing SUPPORTED would inflate the score with facts nobody checked.
    """
    for attempt in range(4):
        try:
            reply = answer_module._client().chat(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE},
                    {
                        "role": "user",
                        "content": f"REQUIRED FACT:\n{fact}\n\nOUR ANSWER:\n{written}\n\nVerdict:",
                    },
                ],
                # Same reason as the pipeline: a judge that marks the
                # same fact differently on a re-run is not a measurement.
                options=answer_module.OPTIONS,
            )
            return "SUPPORTED" in reply["message"]["content"].upper()
        except Exception as exc:
            if attempt == 3:
                print(f"    ! judge failed after 4 tries: {type(exc).__name__}", flush=True)
                return False
            time.sleep(2.0 * (attempt + 1))
    return False


def snapshot(model, by_type, detail, latency) -> dict:
    """The report as it stands. Written after every answer, not just at the end.

    A full run is hours of paid model calls, and this used to be written once
    after the last one -- so a run stopped at hour five produced nothing at
    all, and the only way to get a number was to let the whole thing finish
    uninterrupted. Every partial run is now a usable measurement of however
    many questions it got through.
    """
    return {
        "model": model,
        "answers": len(detail),
        "facts": sum(b["facts"] for b in by_type.values()),
        "supported": sum(b["supported"] for b in by_type.values()),
        "perfect": sum(b["perfect"] for b in by_type.values()),
        "median_seconds": round(statistics.median(latency), 1) if latency else 0,
        "complete": False,
        "by_type": {t: dict(b) for t, b in by_type.items()},
        "detail": detail,
    }


def run(limit: int, types: list[str] | None, model: str, resume: bool = False) -> None:
    asker = Asker()
    pool = [g for g in load_gold() if g.get("answer_facts")]
    if types:
        pool = [g for g in pool if g.get("question_type") in types]

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        buckets[row.get("question_type", "unknown")].append(row)
    per = max(1, limit // max(len(buckets), 1))
    graded = [row for rows in buckets.values() for row in rows[:per]]

    by_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "facts": 0, "supported": 0, "perfect": 0, "abstained": 0}
    )
    detail: list[dict] = []
    latency: list[float] = []

    # `--resume` picks up an interrupted run rather than re-paying for every
    # answer it already graded.
    done: set[str] = set()
    if resume and REPORT.exists():
        try:
            prior = json.loads(REPORT.read_text())
            detail = prior.get("detail", [])
            done = {row["question_id"] for row in detail if not row.get("error")}
            for row in detail:
                bucket = by_type[row["type"]]
                bucket["n"] += 1
                bucket["facts"] += row["facts"]
                bucket["supported"] += row["supported"]
                bucket["perfect"] += row["supported"] == row["facts"]
                bucket["abstained"] += row.get("abstained", 0)
                latency.append(row.get("seconds", 0))
            print(f"resuming — {len(done)} already graded", flush=True)
        except Exception:
            detail, done, latency = [], set(), []
    graded = [row for row in graded if row["question_id"] not in done]

    print(f"grading {len(graded)} answers across {len(buckets)} types\n", flush=True)

    for i, row in enumerate(graded, start=1):
        question = row["question"]
        qtype = row.get("question_type", "unknown")
        t0 = time.time()
        written, failed = "", False
        # A network blip while answering used to be caught and scored as an
        # empty answer -- zero facts supported, indistinguishable from a
        # genuine miss. That is exactly how a rate-limited run comes back
        # looking like a bad system. Retry, and if it still fails, mark the row
        # so it is re-run on the next `--resume` rather than counted.
        for attempt in range(3):
            try:
                written = asker.ask(question).text
                break
            except Exception as exc:
                if attempt == 2:
                    failed = True
                    print(f"  [{i}] pipeline error: {type(exc).__name__}: {exc}",
                          flush=True)
                else:
                    time.sleep(2.0 * (attempt + 1))
        latency.append(time.time() - t0)
        # An empty answer is a broken pipeline, not an abstention. When the
        # model account runs out of quota the request does not raise -- the
        # answer path catches it, degrades, and returns "". Scored as written
        # that is zero facts supported, which is indistinguishable from a
        # system that answered and got everything wrong. It is how a run comes
        # back reporting 0% on six categories that measured 60% and 100% an
        # hour earlier. A genuine abstention says so in words and is not empty.
        if not written.strip():
            failed = True
            print(f"  [{i}] empty answer — pipeline degraded, not scored", flush=True)
        if failed:
            continue

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
                "abstained": int(
                    answer_module.NOT_FOUND in written or not written.strip()
                ),
            }
        )
        print(
            f"  [{i:>3}/{len(graded)}] {qtype:24s} {supported}/{len(facts)} facts"
            f"  {latency[-1]:5.1f}s",
            flush=True,
        )
        # Checkpoint. Cheap next to the model call that produced the row, and
        # the difference between a stopped run being a measurement and being
        # nothing.
        STATE.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(snapshot(model, by_type, detail, latency), indent=2))

    total_facts = sum(b["facts"] for b in by_type.values())
    total_supported = sum(b["supported"] for b in by_type.values())
    total_perfect = sum(b["perfect"] for b in by_type.values())
    n = len(detail)

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
    final = snapshot(model, by_type, detail, latency)
    final["complete"] = True
    REPORT.write_text(json.dumps(final, indent=2))
    print(f"\n  wrote {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=33, help="answers to grade")
    ap.add_argument("--types", help="comma-separated question types")
    ap.add_argument("--model", default=None, help="judge model")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run instead of restarting")
    args = ap.parse_args()
    run(
        args.limit,
        args.types.split(",") if args.types else None,
        args.model or answer_module.ADJUDICATION_MODEL,
        args.resume,
    )


if __name__ == "__main__":
    main()
