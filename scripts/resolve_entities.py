#!/usr/bin/env python
"""Session 3 — entity resolution over the whole normalized corpus.

    python scripts/resolve_entities.py                 # everything
    python scripts/resolve_entities.py --source slack  # one shard
    python scripts/resolve_entities.py --threshold 0.8

Writes `data/state/entities.jsonl` (one canonical entity per line, carrying
its aliases and the reason behind every merge) and `data/state/resolve_stats.json`.
Both are inputs to the review UI, which is where the precision number comes from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse import priors as priors_mod  # noqa: E402
from glasshouse.config import NORMALIZED, STATE  # noqa: E402
from glasshouse.corpus import SOURCES  # noqa: E402
from glasshouse.resolve import MERGE_THRESHOLD, MIN_OCCURRENCES, load_index, resolve  # noqa: E402

ENTITIES = STATE / "entities.jsonl"
STATS = STATE / "resolve_stats.json"
PRIORS = STATE / "priors.json"


def learn_priors(shards: list[Path]) -> priors_mod.Priors:
    """First pass: read the corpus with no assumptions and let it teach us.

    Deliberately separate from the main index build, and deliberately first.
    Qualifier stripping is one of the things being learned, so the names fed
    in here must be raw — normalising them with priors we have not derived yet
    would be assuming the answer.
    """
    emails: list[str] = []
    names: Counter = Counter()
    for shard in shards:
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                emails.extend(doc.get("emails") or ())
                for pair in doc.get("named_emails") or ():
                    if name := pair.get("name"):
                        names[name] += 1
                    if email := pair.get("email"):
                        emails.append(email)
                for speaker in doc.get("speakers") or ():
                    if " " in speaker:
                        names[speaker] += 1
                for att in doc.get("attendees") or ():
                    if name := att.get("name"):
                        names[name] += 1
    return priors_mod.derive(emails, names.items())


def run(sources: list[str], threshold: float, min_occurrences: int) -> None:
    shards = [NORMALIZED / f"{s}.jsonl" for s in sources]
    shards = [p for p in shards if p.exists()]
    if not shards:
        raise SystemExit("no normalized shards found; run scripts/intake.py first")

    t0 = time.time()
    print(f"learning corpus priors from {len(shards)} shards ...", flush=True)
    priors = learn_priors(shards)
    ev = priors.evidence
    print(f"  employer   : {sorted(priors.home_labels)}", flush=True)
    print(f"               top candidates {ev['home_org_candidates'][:4]}", flush=True)
    print(
        f"  qualifiers : {ev['qualifier_count']} learned, e.g. "
        f"{[r[0] for r in ev['qualifier_tokens'][:12]]}",
        flush=True,
    )
    print(
        f"  role inbox : {ev['functional_count']} learned, e.g. "
        f"{[r[0] for r in ev['functional_localparts'][:10]]}",
        flush=True,
    )
    STATE.mkdir(parents=True, exist_ok=True)
    PRIORS.write_text(json.dumps(priors.to_dict(), indent=2, default=list))

    print(f"\nmining identity surfaces ({time.time()-t0:.1f}s so far) ...", flush=True)
    index = load_index(
        shards,
        priors,
        on_progress=lambda n: print(f"  {n:,} docs", end="\r", flush=True),
    )
    print(
        f"  {index.docs_seen:,} docs -> {len(index.surfaces):,} raw surfaces, "
        f"{len(index.hard_links):,} stated name<->email bindings  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    shared = index.shared_emails()
    links = index.exclusive_links()
    working = index.working_set(min_occurrences)
    print(
        f"  {len(shared):,} shared mailboxes excluded; "
        f"{len(links):,} of {len(index.hard_links):,} stated bindings are exclusive",
        flush=True,
    )
    print(f"  working set: {len(working):,} surfaces", flush=True)

    t1 = time.time()
    entities, diag = resolve(working, links, threshold)
    diag["docs"] = index.docs_seen
    diag["raw_surfaces"] = len(index.surfaces)
    diag["stated_bindings"] = len(index.hard_links)
    diag["exclusive_bindings"] = len(links)
    diag["shared_mailboxes_excluded"] = len(shared)
    diag["rejected_values"] = dict(index.rejected)
    diag["min_occurrences"] = min_occurrences
    diag["priors"] = priors.evidence
    diag["sources"] = sources
    diag["resolve_seconds"] = round(time.time() - t1, 1)

    STATE.mkdir(parents=True, exist_ok=True)
    with ENTITIES.open("w", encoding="utf-8") as fh:
        for e in entities:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    STATS.write_text(json.dumps(diag, indent=2))

    print(f"\n{'='*70}")
    print(f"  candidate pairs      {diag['candidate_pairs']:>10,}")
    print(f"  above threshold      {diag['pairs_above_threshold']:>10,}")
    print(f"  merges applied       {diag['merges_applied']:>10,}")
    print(f"  refused (constraint) {diag['merges_refused_by_constraint']:>10,}")
    print(f"  entities             {diag['entities']:>10,}")
    print(f"  multi-alias entities {diag['multi_alias_entities']:>10,}")
    if diag["blocks_dropped_oversized"]:
        print(
            f"  NOTE {diag['blocks_dropped_oversized']} blocks dropped as oversized "
            f"(largest: {diag['largest_dropped_blocks'][:3]})"
        )
    print("\n  signals fired:")
    for sig, n in diag["signal_counts"].items():
        print(f"    {sig:22s} {n:>8,}")
    print(f"\n  wrote {ENTITIES} in {time.time()-t0:.1f}s total")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", help="restrict to source(s)")
    ap.add_argument("--threshold", type=float, default=MERGE_THRESHOLD)
    ap.add_argument("--min-occurrences", type=int, default=MIN_OCCURRENCES)
    args = ap.parse_args()
    run(args.source or list(SOURCES), args.threshold, args.min_occurrences)


if __name__ == "__main__":
    main()
