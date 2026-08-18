# Glasshouse Handoff — Phase 5 onward

Continue building Glasshouse (HackHydra **Track 01: enterprise context and
ontology**, deadline **2026-08-20**). `IMPLEMENTATION_PLAN.md` has the phase
structure, but several of its assumptions are now disproven by measurement.
**The corrections below take precedence over the plan.**

## What the project is

Half a million documents from nine enterprise sources (Slack, Gmail, Linear,
Google Drive, HubSpot, Fireflies, GitHub, Jira, Confluence) turned into a
queryable ontology in HydraDB, answering questions from simple lookups through
multi-hop reasoning, conflict resolution, and recognising when the answer is
absent. Judges weight technical execution, use of HydraDB and graph-native
approaches, product completeness, quality of results, and originality, and say
they care about "working, thoughtful products, not just benchmark scores".
There is a separate $500 "Best Use of HydraDB" award for a strong graph model
and a use case hard to do with vector or relational stores.

## State

Everything is loaded and running. `docker compose up -d` starts the engine
(container `glasshouse-engine`). 511,962 documents in a 4GB SQLite FTS index;
an ontology of 209,388 surface forms collapsing to 39,847 resolved people;
`(:Entity)-[:MENTIONED_IN]->(:Document)` across the whole corpus. 106 tests
pass. Retrieval is 0.25–0.96s; a full question is ~3s end to end.

```bash
.venv/bin/python -m uvicorn glasshouse.server:app --host 127.0.0.1 --port 8080 --app-dir src
```

## Measured findings — do not re-derive these

1. **Person-seeded graph retrieval does not work on this benchmark.** Only
   **21 of 570** questions name a resolvable person (0 of 20
   `conflicting_info`, 0 of 30 `constrained`, 0 of 125 `semantic`). Where it
   did add documents it added 10–20 and **never a correct one**. The plan's
   Phase 4 A/B recall work targets ~4% of the benchmark and wins on none of
   it. Do not spend time tuning that path.

2. **The graph has a second entrance that does work.** `who owns X` names
   nobody — the person is the answer. `GraphEngine.entities_for_documents`
   reads `MENTIONED_IN` **inwards**, from the documents retrieval already
   found to the people attached to them, ranked by how many of those
   documents each connects to. 0.03–0.06s per document. This fires on
   effectively every question. It has changed answers (a "who owns the
   audit-log shipper sidecar" question went from naming nobody to naming a
   person) but cannot establish ownership, because co-occurrence is not
   ownership and the prompt correctly forbids the model from pretending
   otherwise.

3. **The biggest answer-quality win so far was not the graph.** `DOC_CHARS`
   truncated every document to its first 2600 characters while corpus
   documents run 5–7k, so answering sentences were routinely cut off — the
   burst-credits conflict answer sat at character 4169.
   `answer.select_passages` now scores windows across the document weighted
   toward rare terms. That flipped the question from "the documents do not
   specify" to the exact correct answer, and generalised to untested ones.

4. **Retrieval is not the bottleneck for conflicts.** Plain FTS retrieves the
   expected document on **10/10** `conflicting_info` questions, ranked #1 or
   #2 in 9/10, and it reaches the synthesis context. The model still scores
   ~48% because it cannot adjudicate between competing values it can see.
   That is the case for Phase 5 claims and arbitration.

5. **`metadata` scores 0%** across 100 benchmark questions. These ask about
   authorship, ownership, and location. The graph stores `MENTIONED_IN` but
   **not authorship**. Adding author/owner edges is the most likely way to
   make HydraDB genuinely load-bearing, and the reverse traversal above is
   the machinery it would run on.

## The baseline is stale

`scripts/grade.py` grades answers against the benchmark's `answer_facts`
rubric with an LLM judge, per fact, stratified by category — the only
measurement of answer quality that exists; `scripts/score.py` measures
document recall only. Its one recorded run reported **35.5% fact recall,
18.2% fully correct** on 44 questions.

**That run predates passage selection, the identity gate, the reverse
traversal and the performance fix. Re-run it before quoting any number:**

```bash
.venv/bin/python scripts/grade.py --limit 44
```

Per-category figures come from 4 questions each and two of them were wrong
when spot-checked by hand. Treat them as directional only.

## Performance notes worth keeping

- `docs` is an FTS5 virtual table with `doc_id UNINDEXED`, so `WHERE doc_id
  IN (...)` scanned all 511,962 rows — **21.5s** to fetch the 55 documents the
  graph had selected. A `docmap(doc_id, rid)` side table plus rowid lookups
  made it **0.01s**. `build_index.py` builds it; `LocalRecall.build_docmap()`
  rebuilds it after the index changes.
- `search_scoped` ranks a bounded scope in Python rather than through FTS.
  Two FTS approaches each cost ~25s because SQLite evaluates `docs MATCH`
  against the whole index before joining to the scope. Safe only because the
  caller caps the scope; not a general search path.
- A labeled `MATCH` expansion in HydraDB scans the whole `MENTIONED_IN` edge
  set: 15–30s per seed, often past the 30s cap, even with `LIMIT 1`.
  `algo.SSpaths` returns 200 paths in 0.04–0.27s. Do not reintroduce `MATCH`.
- SQLite connections are thread-local. One process-wide `Asker` on FastAPI's
  threadpool raised "SQLite objects created in a thread can only be used in
  that same thread" intermittently.

## Rules

- **Gold answers** live at `$GOLD_ANSWERS_PATH` outside the repo. Only
  `scripts/score.py` and `scripts/grade.py` may read them. Nothing under
  `src/glasshouse` may import or see them.
- Commits: subject line only, no body, never mention Claude or AI.
- HydraDB rejects unanchored global scans with `429`/timeout. Keep queries
  anchored and bounded.
- The user tests the UI themselves — hand over the URL, do not drive Chrome.
- Do not claim a graph-native retrieval win in README, UI or video. The
  measurement does not support it. The UI currently states honestly when the
  graph added nothing.
- The trace must describe what happened. Two labels have already had to be
  fixed for overstating ("searched the corpus 20 docs" when 134,466 matched;
  "hydradb reached from 0 entities" when the other entrance had opened).

## Suggested order

1. Re-run `scripts/grade.py` for a current baseline.
2. Diagnose `metadata` 0% and add authorship/ownership edges if that is the
   cause — the highest-value remaining graph work.
3. Phase 5 claims, contradiction and arbitration.
4. Phase 6/7: demo polish, README, video, submission. Reserve time; do not
   let this get squeezed. Prefer a hosted URL plus a small prebuilt sample
   corpus over a local installer that starts a multi-hour ingest.

Demo question that exercises both graph entrances:
*"In the customer success shared drive, which draft spreadsheet owned by
Jordan Reyes was last modified most recently?"* — 1 seed entity, 55 candidate
documents, ablation keyword 20 / identity 20 / graph 20 with 11–12 graph-only,
and connected entities Jordan Reyes (5 docs) and Priya Nair (2).
