# Glasshouse Handoff — after the Phase 3/4 correction

Continue building Glasshouse (HackHydra Track 01, deadline **2026-08-20**).
Read `IMPLEMENTATION_PLAN.md` for the phase structure, but **parts of it are
now known to be wrong** — the corrections are below and they take precedence.

## What the project is

Half a million documents from nine enterprise sources (Slack, Gmail, Linear,
Google Drive, HubSpot, Fireflies, GitHub, Jira, Confluence) turned into a
queryable ontology in HydraDB, answering questions that range from lookups to
multi-hop reasoning, conflict resolution, and knowing when the answer is
absent. Judges weight: technical execution, use of HydraDB and graph-native
approaches, product completeness, quality of results, originality — and state
outright that they care about "working, thoughtful products, not just
benchmark scores." There is a separate $500 "Best Use of HydraDB" award for a
strong graph model and a use case hard to do with vector or relational stores.

## State

Phases 0-3 are implemented. The engine runs locally via
`docker compose up -d` (container `glasshouse-engine`). Everything is loaded:
511,962 documents in a 4GB SQLite FTS index, an ontology of 209,388 surface
forms collapsing to 39,847 resolved people, and `(:Entity)-[:MENTIONED_IN]->
(:Document)` edges across the whole corpus. 101 tests pass. Server:

```bash
.venv/bin/python -m uvicorn glasshouse.server:app --host 127.0.0.1 --port 8080 --app-dir src
```

## Findings that change the plan — do not re-derive these

1. **Person-seeded graph retrieval does not work on this benchmark.** Only
   **21 of 570** questions name a resolvable person at all (0 of the 20
   `conflicting_info`, 0 of 30 `constrained`, 0 of 125 `semantic`). Of the
   questions where the graph did contribute documents, it added 10-20 each and
   **never** added a correct one. The Phase 3 gate fails on real data. Stop
   tuning this path; the plan's Phase 4 A/B recall work is aimed at ~4% of the
   benchmark and wins on none of it.

2. **Answer quality was never measured before now.** `scripts/score.py` only
   ever measured document recall. `scripts/grade.py` (new) grades answers
   against the benchmark's `answer_facts` rubric with an LLM judge, per fact,
   stratified by category. First baseline, 44 questions:
   **35.5% fact recall, 18.2% fully correct**, median 2.0s, p95 2.8s.
   Per category: `info_not_found` 100%/100% (abstention genuinely works),
   `metadata` 0%, `miscellaneous` 0%, everything else 24-50% fact recall at 0%
   fully correct. Only 4 questions per category, so treat per-category numbers
   as directional — two of them were wrong when spot-checked by hand.

3. **The biggest quality win was passage selection, not the graph.**
   `DOC_CHARS = 2600` truncated every document to its head while corpus
   documents run 5-7k characters, so answering sentences were routinely cut
   off. `answer.select_passages` now scores windows across the document
   weighted toward rare terms. This flipped the burst-credits conflict
   question from "the documents do not specify" to the exact correct answer,
   and generalised to an untested conflict question.

4. **Retrieval is not the bottleneck for conflicts.** Plain FTS retrieves the
   expected document on **10/10** `conflicting_info` questions, ranked #1 or #2
   in 9/10, and it reaches the synthesis context. The model still scores ~48%
   because it cannot adjudicate between competing values it can see.

## Where the graph should earn its place

Not document discovery. Two candidates, both evidence-backed:

- **Claims and contradictions (Phase 5).** Docs are already retrieved; what is
  missing is `(:Claim)` nodes with `CONTRADICTS`/`SUPERSEDES` edges, trust
  scoring, and a winner with a rationale. Directly named by the track as a hard
  part, has its own 20-question category, awkward in vector/relational stores.
- **`metadata` questions at 0%** (100 questions in the benchmark). These ask
  about authorship, ownership, and location — structural facts. The graph
  stores `MENTIONED_IN` but **not authorship**. Adding author/owner edges is a
  plausible fix and worth diagnosing before building.

## Changed this session (uncommitted)

- `graph.py` — `documents_for_entities` now uses `algo.SSpaths`. A labeled
  `MATCH` expansion scans the whole edge set: 15-30s per seed and often past
  the engine's 30s cap even with `LIMIT 1`. SSpaths returns 200 paths in
  0.04-0.27s. Do not reintroduce the MATCH form.
- `ask.py` — identity gating: personhood must be demonstrated (a name alias
  plus a second spelling or cross-kind corroboration); `_organizational()`
  rejects entities whose name tokens appear in their own mail domain;
  `_metric_shaped()` rejects `p95`/`429`. Killed `sre`, `finance`, `support`,
  `Acme Health`, `Horizon Analytics` as "people".
- `answer.py` — `select_passages()` plus a stopword list.
- `recall.py` + `ask.py` — SQLite connections are now **thread-local**. One
  process-wide `Asker` on FastAPI's threadpool was raising "SQLite objects
  created in a thread can only be used in that same thread" intermittently.
- `web/index.html` — retrieval A/B panel rendering the `graph_scope`,
  `graph_document`, `graph_ablation` events the backend already emitted; header
  now says "surface forms / resolved people" rather than claiming 166k people.
- New: `scripts/grade.py`, `tests/test_phase4_passages.py`,
  `tests/test_thread_safety.py`.

## Rules

- **Gold answers** live at `$GOLD_ANSWERS_PATH` outside the repo. Only
  `scripts/score.py` and `scripts/grade.py` may read them. Nothing under
  `src/glasshouse` may import or see them.
- Commits: subject line only, no body, and never mention Claude or AI.
- HydraDB rejects unanchored global scans with `429`/timeout. Keep queries
  anchored and bounded.
- The user tests the UI themselves — hand over the URL, do not drive Chrome.
- Do not claim a graph-native retrieval win anywhere (README, video, UI). The
  measurement does not support it.

## Suggested next step

Diagnose `metadata` 0% first — it is 100 benchmark questions and the likely fix
(authorship edges) would finally make the graph load-bearing. Then Phase 5
claims and arbitration. Hold the one-click setup / hosted deploy for Phase 7,
and prefer a hosted URL plus a small prebuilt sample corpus over a local
installer that starts a multi-hour ingest.
