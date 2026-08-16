# Phase 3 Agent Handoff: Graph-Scoped Document Retrieval

Implement **Phase 3 only** from `IMPLEMENTATION_PLAN.md`. Do not start Phase 4
unless the Phase 3 gate passes.

## Current State

- Phase 0 passed. The verified local HydraDB image supports:

  ```cypher
  CALL algo.SPpaths({
    sourceNode: $source,
    targetNode: $target,
    relTypes: ['RELATES'],
    relDirection: 'both',
    maxLen: 3,
    pathCount: 5
  })
  YIELD path, pathWeight, pathCost
  RETURN path, pathWeight, pathCost
  ```

- The raw HTTP response shape is recorded at
  `tests/fixtures/hydradb_sp_paths_response.json`.
- `GraphEngine.query(...)` unwraps typed scalar cells and path properties.
- `GraphEngine.paths(...)` currently uses the verified `SPpaths` form.
- HydraDB runs locally through Docker, not through HydraDB Cloud.
- Phase 1 passed. Do not reintroduce substring identity matching or claims that
  a shared document proves collaboration, ownership, agreement, or a decision.
- Phase 2 completed its full loader run over all `511,962` normalized records.
  The checkpoint is `data/state/document_graph_checkpoint.json` at the final
  Slack record. A no-op rerun reports zero new writes.
- The offline graph has `(:Entity)-[:MENTIONED_IN]->(:Document)` edges created
  only from structured normalized identity fields. Document bodies are not in
  HydraDB.
- The graph has more than HydraDB's global count admission limits: `MATCH`
  count scans over all `Document` nodes or all `MENTIONED_IN` edges receive
  `429 resource_exhausted`. Use bounded, anchored queries rather than global
  scans. An anchored Entity-to-Document connection was verified successfully.

## Required Reading

Read all of these before editing:

1. Phase 3 in `IMPLEMENTATION_PLAN.md`.
2. `tests/fixtures/hydradb_sp_paths_response.json`.
3. `src/glasshouse/graph.py`.
4. `src/glasshouse/recall.py`.
5. `src/glasshouse/ask.py`.
6. `scripts/score.py`.
7. Existing tests, especially `tests/test_phase1_correctness.py` and
   `tests/test_load_document_graph.py`.

## Allowed Files

- `src/glasshouse/graph.py`
- `src/glasshouse/recall.py`
- `src/glasshouse/ask.py`
- Focused files under `tests/`

Do not modify the Phase 2 loader in this phase.

## Required Implementation

Implement and retain all three retrieval strategies separately:

```text
plain_docs     = FTS(question)
identity_docs  = FTS(topic AND aliases of resolved query entities)
graph_scope    = documents directly reachable from resolved entity node IDs
graph_docs     = topic-ranked documents restricted to graph_scope
final_docs     = dedupe(identity_docs + graph_docs), preserving provenance
```

Requirements:

- Begin with direct `Entity -> Document` neighbors only. Do not add
  collaborator or broad multi-hop expansion until direct graph retrieval wins.
- Add the specified typed `GraphCandidate` and `RetrievalResult` shapes.
- Add `LocalRecall.get_many(doc_ids)` preserving requested order.
- Topic-rank only a bounded graph candidate set. Do not load thousands of
  document bodies into Python for a custom scorer.
- Use one `Asker.retrieve(question, limit)` path for streaming and non-streaming
  callers.
- Stop query-time canonical-document graph writes in `Asker.connect`; Phase 2
  owns graph linking now.
- Emit `graph_scope`, `graph_document`, and `graph_ablation` events.
- Treat graph reachability as candidate provenance, never as proof of an answer
  or a relationship semantic.
- Preserve FTS rank separately from graph provenance.

## Tests And Gate

Create the Phase 3 fixture described in the plan, including:

- A question naming `Sam`.
- Plain FTS cannot retrieve the answer document.
- Ontology resolves `Sam` and `S. Ratnaparkhi` to one entity.
- The offline graph reaches the answer document.
- Graph-scoped topic ranking returns it with seed and path provenance.
- Topic relevance outranks unrelated co-mentioned documents.
- No named entity falls back to plain FTS only.

Run exactly:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/ask.py --trace "What did Sam decide about the retention policy?"
```

Do not proceed to Phase 4 unless a reproducible real-corpus question satisfies:

```text
expected document absent from plain top 20
expected document absent from identity top 20, or materially lower
expected document present in graph-aware top 20
final answer cites the graph-added document
```

If that gate fails, stop and inspect graph coverage and ranking. Do not begin
claims, conflicts, or UI work.
