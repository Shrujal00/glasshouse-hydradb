#!/usr/bin/env python
"""Session 2 — push the normalized corpus into HydraDB cloud (the recall layer).

    python scripts/ingest_cloud.py                  # everything, resuming
    python scripts/ingest_cloud.py --limit 20       # smoke test
    python scripts/ingest_cloud.py --source slack   # one source

Runs for hours over 512k documents, so it is built to be interrupted: progress
is checkpointed per source and a rerun picks up where it stopped. Ingest is
`upsert=true`, so a replayed batch overwrites rather than duplicating - a crash
mid-batch costs nothing but the time to send it again.

Two things the API taught us the hard way, both encoded below:

- **There is a rate limit** despite the tier advertising none: 1000 ingestion
  items per minute, answered with a 429 carrying `Retry-After`. Unthrottled
  concurrency does not go faster, it just converts documents into dropped
  batches. So the sender is paced to stay under the limit rather than
  discovering it on every request.
- **A checkpoint must track the contiguous prefix**, not a running total. With
  concurrent batches, counting completions lets a *failed* batch be stepped
  over by the successes behind it, and those documents would then never be
  sent again. Only the unbroken run from the start is safe to skip on resume.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glasshouse.cloud import CloudRecall, batched  # noqa: E402
from glasshouse.config import NORMALIZED, STATE  # noqa: E402
from glasshouse.corpus import SOURCES  # noqa: E402

CHECKPOINT = STATE / "ingest_checkpoint.json"

# One request carries this many documents as multipart files. Average document
# is ~5 KB, so 64 keeps a request near 350 KB.
BATCH = 64

# The observed ceiling is 1000 items/minute; sitting just under it keeps the
# pipe full without spending attempts on 429s.
ITEMS_PER_MIN = 900

# A handful of documents are pathologically long. The recall layer chunks
# server-side anyway, so truncating bounds request size without losing the
# retrievable head of the document.
MAX_CHARS = 200_000

_RETRY_AFTER = re.compile(r"retry in (\d+) second", re.I)


class RateLimiter:
    """Token bucket shared by every upload thread."""

    def __init__(self, per_minute: int, burst: int = BATCH * 3) -> None:
        # Capacity is the *burst* allowance, deliberately far below the
        # per-minute rate. A bucket that starts full at 900 lets fourteen
        # batches leave at once; the server counts those against a short
        # window and 429s most of them, and with many workers the resulting
        # penalties overlap until throughput collapses. Observed: 20 workers
        # with a 900-item burst ran at 0.1 docs/s, slower than 3 workers.
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.rate = per_minute / 60.0
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, n: int) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.updated) * self.rate
                )
                self.updated = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))

    def penalize(self, seconds: float) -> None:
        """Drain the bucket after a 429, so every thread backs off together.

        The deferral is not allowed to stack. Several workers usually get the
        same 429 at the same moment, and pushing the refill clock forward once
        per report compounds a 60-second penalty into minutes of dead air.
        """
        with self.lock:
            self.tokens = 0.0
            self.updated = max(self.updated, time.monotonic() + seconds)


class Checkpoint:
    """Per-source count of documents confirmed sent, as a contiguous prefix."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.done: dict[str, int] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, source: str) -> int:
        return self.done.get(source, 0)

    def set(self, source: str, n: int) -> None:
        with self.lock:
            if n <= self.done.get(source, 0):
                return
            self.done[source] = n
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.done, indent=2))
            tmp.replace(self.path)  # atomic: a kill mid-write cannot corrupt it


def iter_docs(source: str, skip: int, limit: int | None):
    """Yield normalized records from one shard, skipping what is already sent."""
    path = NORMALIZED / f"{source}.jsonl"
    if not path.exists():
        return
    sent = 0
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < skip:
                continue
            if limit is not None and sent >= limit:
                return
            doc = json.loads(line)
            if len(doc.get("body") or "") > MAX_CHARS:
                doc["body"] = doc["body"][:MAX_CHARS]
            yield doc
            sent += 1


def count_lines(source: str) -> int:
    path = NORMALIZED / f"{source}.jsonl"
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def send(recall: CloudRecall, limiter: RateLimiter, batch: list[dict], attempts: int = 6) -> bool:
    """Upload one batch, pacing ahead of the limit and obeying it when told to.

    Returns False only when a batch is rejected for a reason retrying will not
    fix, so one malformed document cannot cost the rest of the corpus.
    """
    attempt = 0
    throttled = 0
    while attempt < attempts:
        limiter.acquire(len(batch))
        try:
            recall.ingest(batch)
            return True
        except Exception as exc:
            text = str(exc)
            if "RATE_LIMITED" in text or "429" in text:
                # Being throttled is expected traffic shaping, not a failure,
                # so it must not consume the error budget - a batch dropped for
                # politeness is a batch of documents nobody can search.
                m = _RETRY_AFTER.search(text)
                delay = float(m.group(1)) if m else 60.0
                limiter.penalize(delay)
                time.sleep(delay)
                throttled += 1
                if throttled > 20:
                    print("    ! batch dropped: throttled 20 times", flush=True)
                    return False
                continue
            attempt += 1
            if attempt >= attempts:
                print(f"    ! batch dropped: {type(exc).__name__}: {text[:200]}", flush=True)
                return False
            time.sleep(2**attempt)
    print("    ! batch dropped after exhausting retries on rate limiting", flush=True)
    return False


def run(sources: list[str], limit: int | None, workers: int) -> None:
    recall = CloudRecall()
    limiter = RateLimiter(ITEMS_PER_MIN)
    print(f"database {recall.database}: {recall.ensure_database()}", flush=True)
    try:
        print(f"already indexed: {recall.indexed_count():,} rows", flush=True)
    except Exception as exc:
        print(f"(stats unavailable: {exc})", flush=True)

    ckpt = Checkpoint(CHECKPOINT)
    started = time.time()
    grand = 0

    for source in sources:
        total = count_lines(source)
        skip = ckpt.get(source)
        if not total:
            print(f"  {source:14s} no shard, skipping", flush=True)
            continue
        if skip >= total and limit is None:
            print(f"  {source:14s} {total:,} already sent", flush=True)
            continue

        print(f"  {source:14s} {total - skip:,} to send (resuming at {skip:,})", flush=True)
        t0 = time.time()
        # Batch index -> documents in it, for batches that came back accepted.
        accepted: dict[int, int] = {}
        pending: list[tuple[int, int, object]] = []
        failures = 0
        frontier = 0  # next batch index still missing from the landed prefix
        prefix_docs = 0

        def retire(upto: int) -> None:
            """Fold finished futures into the checkpoint, prefix-first."""
            nonlocal frontier, failures, prefix_docs
            while pending and len(pending) > upto:
                idx, size, fut = pending.pop(0)
                if fut.result():
                    accepted[idx] = size
                else:
                    failures += 1
            while frontier in accepted:
                prefix_docs += accepted.pop(frontier)
                frontier += 1
            ckpt.set(source, skip + prefix_docs)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, batch in enumerate(batched(iter_docs(source, skip, limit), BATCH)):
                pending.append((idx, len(batch), pool.submit(send, recall, limiter, batch)))
                # Keep the queue shallow: an interrupt should strand a couple of
                # batches, not a thousand.
                retire(workers * 2)
                done = ckpt.get(source)
                if idx and idx % 40 == 0:
                    rate = (done - skip) / max(time.time() - t0, 1e-6)
                    eta = (total - done) / max(rate, 1e-6) / 60
                    print(
                        f"    {source} {done:>7,}/{total:,}  {rate:5.1f} docs/s  "
                        f"eta {eta:6.1f}m",
                        flush=True,
                    )
            retire(0)

        landed = ckpt.get(source) - skip
        grand += landed
        dt = time.time() - t0
        note = f"  ({failures} batches dropped)" if failures else ""
        print(
            f"  {source:14s} sent {landed:,} in {dt/60:.1f}m "
            f"({landed/max(dt,1e-6):.1f}/s){note}",
            flush=True,
        )

    print(f"\n{'='*70}", flush=True)
    print(f"queued {grand:,} documents in {(time.time()-started)/60:.1f}m", flush=True)
    try:
        print(f"indexed so far: {recall.indexed_count():,} rows", flush=True)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max docs per source")
    ap.add_argument("--source", action="append", help="restrict to source(s)")
    ap.add_argument("--workers", type=int, default=3, help="concurrent upload threads")
    ap.add_argument(
        "--smallest-first",
        action="store_true",
        help="order sources by shard size, so every source is represented early",
    )
    args = ap.parse_args()
    sources = args.source or list(SOURCES)
    if args.smallest_first:
        sources.sort(key=count_lines)
    run(sources, args.limit, args.workers)


if __name__ == "__main__":
    main()
