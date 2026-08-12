#!/usr/bin/env python
"""Session 1 — corpus intake.

Walks the extracted EnterpriseRAG-Bench corpus, parses every document into the
normalized shape, and writes one JSONL shard per source plus a stats summary.

    python scripts/intake.py [--limit N] [--source slack]

Runs across cores because it is 512k small files and purely I/O plus regex.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.corpus import SOURCES, iter_source_files, parse_document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / "normalized"


def _parse_one(args: tuple[str, str]) -> dict | None:
    path_str, source = args
    try:
        return parse_document(Path(path_str), source, CORPUS).to_dict()
    except Exception as exc:  # keep going; report at the end
        return {"__error__": f"{type(exc).__name__}: {exc}", "__path__": path_str}


def run(sources: list[str], limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grand = Counter()
    errors: list[str] = []
    started = time.time()

    for source in sources:
        files = list(iter_source_files(CORPUS, source))
        if limit:
            files = files[:limit]
        if not files:
            print(f"  {source:14s} no files, skipping")
            continue

        t0 = time.time()
        stats = Counter()
        out_path = OUT / f"{source}.jsonl"

        with mp.Pool(processes=min(mp.cpu_count() - 2, 16)) as pool, out_path.open(
            "w", encoding="utf-8"
        ) as fh:
            work = ((str(p), source) for p in files)
            for rec in pool.imap_unordered(_parse_one, work, chunksize=256):
                if rec is None:
                    continue
                if "__error__" in rec:
                    errors.append(f"{rec['__path__']}: {rec['__error__']}")
                    stats["errors"] += 1
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["docs"] += 1
                stats["with_email"] += bool(rec["emails"])
                stats["with_named_email"] += bool(rec["named_emails"])
                stats["with_speakers"] += bool(rec["speakers"])
                stats["with_mentions"] += bool(rec["mentions"])
                stats["with_ticket"] += bool(rec["ticket_key"])
                stats["with_date"] += bool(rec["date"])
                stats["emails_total"] += len(rec["emails"])
                stats["named_emails_total"] += len(rec["named_emails"])
                stats["speakers_total"] += len(rec["speakers"])
                stats["mentions_total"] += len(rec["mentions"])

        dt = time.time() - t0
        print(
            f"  {source:14s} {stats['docs']:>7,} docs  {dt:6.1f}s  "
            f"email={stats['with_email']:>6,}  named={stats['with_named_email']:>6,}  "
            f"spk={stats['with_speakers']:>6,}  @={stats['with_mentions']:>6,}  "
            f"tkt={stats['with_ticket']:>6,}"
        )
        for k, v in stats.items():
            grand[k] += v

    print(f"\n{'='*76}\nTOTAL {grand['docs']:,} documents in {time.time()-started:.1f}s")
    print(
        f"  identity signals: {grand['emails_total']:,} emails, "
        f"{grand['named_emails_total']:,} name<->email pairs, "
        f"{grand['speakers_total']:,} speaker turns, {grand['mentions_total']:,} @mentions"
    )
    if errors:
        print(f"  {len(errors)} parse errors (first 5):")
        for e in errors[:5]:
            print("   ", e)

    (OUT / "_stats.json").write_text(json.dumps(dict(grand), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max files per source")
    ap.add_argument("--source", action="append", help="restrict to source(s)")
    args = ap.parse_args()
    run(args.source or list(SOURCES), args.limit)


if __name__ == "__main__":
    main()
