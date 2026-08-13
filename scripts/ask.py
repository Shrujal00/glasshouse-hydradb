#!/usr/bin/env python
"""Ask Glasshouse a question.

    python scripts/ask.py "who owns the audit-log shipper?"
    python scripts/ask.py --blind 5        # 5 real benchmark questions
    python scripts/ask.py --trace "..."    # show the reasoning events

Everything runs locally: the corpus index, the ontology and the HydraDB engine
in Docker. No API key, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.ask import Asker  # noqa: E402
from glasshouse.config import DATA  # noqa: E402

BLIND = DATA / "bench" / "questions_blind.jsonl"


def show(asker: Asker, question: str, trace: bool) -> None:
    answer = asker.ask(question)
    print(f"\n\033[1m? {question}\033[0m")
    if trace:
        for e in answer.events:
            print(f"    · {e.line()}")
    print(answer.render())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the question to ask")
    ap.add_argument("--blind", type=int, help="run N questions from the blind benchmark set")
    ap.add_argument("--offset", type=int, default=0, help="skip N blind questions first")
    ap.add_argument("--trace", action="store_true", help="print reasoning events")
    args = ap.parse_args()

    asker = Asker()

    if args.blind:
        rows = [json.loads(l) for l in BLIND.open(encoding="utf-8")]
        for row in rows[args.offset : args.offset + args.blind]:
            show(asker, row["question"], args.trace)
        return

    if not args.question:
        ap.error("give a question, or --blind N")
    show(asker, " ".join(args.question), args.trace)


if __name__ == "__main__":
    main()
