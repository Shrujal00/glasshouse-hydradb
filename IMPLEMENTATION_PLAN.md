# Glasshouse: Make HydraDB Load-Bearing

**Written:** 2026-08-14

**Deadline:** 2026-08-20, 11:59 PM PT

**Audience:** implementation agents, including lower-capability models

**Goal:** make HydraDB retrieve or adjudicate evidence that plain full-text search cannot, and prove the difference with a reproducible A/B evaluation and demo.

## 1. Current truth

The current query path is:

```text
question
  -> local FTS5 retrieves documents
  -> identities are resolved from those documents
  -> only those already-retrieved documents are written to HydraDB
  -> HydraDB draws co-mention paths
  -> an LLM writes the answer
```

This means HydraDB cannot currently discover a document that FTS5 missed. It is downstream of retrieval. The graph is visible, but it is not load-bearing.

The current identity-expansion experiment also does not establish value:

- saved sample: plain recall `68/120`; identity-aware recall `68/120`;
- role mailboxes such as `Security Lead` and `Marketplace Onboarding` are incorrectly treated as people;
- some benchmark categories lose relevant documents after expansion;
- `MENTIONED_IN` edges use substring matching, so a surface such as `sam` can match `same` or `sample`;
- the answer prompt incorrectly says that co-mention proves collaboration or ownership.

Do not add more UI polish until the graph changes a measured result.

## 2. Product definition

Glasshouse is not “Ctrl+F over everything.” It is:

> A queryable enterprise evidence graph that resolves who people are, retrieves evidence through relationships, preserves conflicting claims, chooses the active claim with an auditable reason, and abstains when the graph cannot support an answer.

The product has three layers, built in this order:

1. **Identity:** aliases resolve to one canonical entity with stored evidence.
2. **Graph retrieval:** a resolved entity can reach relevant documents that keyword retrieval did not return.
3. **Claims and conflicts:** contradictory facts remain in the graph; one may win, but the losing claim and rationale remain inspectable.

The minimum winning demonstration is an A/B sequence:

```text
same question
  plain FTS -> misses the answer document or returns the wrong active fact
  Glasshouse -> HydraDB reaches the evidence path or arbitrates the claims
  UI -> shows the path, sources, and reason
```

## 3. Execution rules for lower-capability models

Every phase is a separate implementation context. Do not begin the next phase until the current phase passes its gate.

For every phase:

1. Read every file listed under **Required reading** before editing.
2. Edit only the files listed under **Allowed files**.
3. Copy existing repository patterns named under **Patterns to copy**.
4. Add tests before or with the implementation.
5. Run every verification command exactly as written.
6. Report files changed, commands run, outputs, and remaining risks.
7. Stop if an API or return shape differs from documentation. Do not invent an alternative silently.
8. Do not commit until a separate verification/review pass succeeds.

Never expose or import gold answers into `src/glasshouse`. Gold data is permitted only in evaluation scripts.

Do not parallel-edit these high-conflict files:

- `src/glasshouse/ask.py`
- `src/glasshouse/graph.py`
- `scripts/load_graph.py`

Assign one implementation owner at a time when a phase touches any of them.

## 4. Phase 0 — verify allowed APIs and freeze schemas

### Objective

Confirm exact HydraDB query syntax and freeze the graph/event schemas before implementation.

### Required reading

- Official HydraDB `README.md`, sections **Querying** and **Native path procedures**.
- Official HydraDB `cypher-compat.md`.
- Official HydraDB examples that invoke `algo.SPpaths`, `algo.SSpaths`, or `algo.MSpaths`.
- `src/glasshouse/graph.py`
- `scripts/load_graph.py`
- `src/glasshouse/corpus.py`
- `docs/superpowers/specs/2026-08-13-glasshouse-design.md`, sections 2–4, if that ignored local design file is present.

### Allowed APIs

Existing project wrappers:

- `GraphEngine.query(cypher, parameters=None, strong=False, timeout=120.0)`
- `GraphEngine.upsert_nodes(label, rows, properties)`
- `GraphEngine.merge_edges(rel, rows, properties, src_label, dst_label=None)`
- `GraphEngine.paths(source, target, rel_types, max_len=3, path_count=5)`
- `LocalRecall.get(doc_id)`
- `LocalRecall.search(question, limit=20, source=None, also=(), drop=())`
- `node_id(key)` for stable non-negative integer IDs

Documented HydraDB multi-path pattern to verify against the current image:

```cypher
CALL algo.MSpaths({
  sourceLabel: 'Entity',
  sourceProperty: 'name',
  sourceValues: ['alpha', 'beta'],
  targetValues: ['alpha', 'beta'],
  pairwise: true,
  relTypes: ['RELATES'],
  relDirection: 'both',
  maxLen: 3,
  pathCount: 5,
  fairRelationshipVariants: true,
  resultLimit: 100
})
YIELD path
RETURN path
```

The implementation must use the exact property names and `YIELD` shape supported by the running HydraDB image. If the checked-in documentation and image disagree, record the probe result in a test or code comment and use the image-supported form.

### Schema to freeze

Existing:

```text
(:Alias)-[:RESOLVES_TO]->(:Entity)
(:Entity)-[:MENTIONED_IN]->(:Document)
```

New retrieval provenance object returned by Python:

```python
GraphCandidate(
    doc_id: str,
    seed_eids: tuple[str, ...],
    path: dict,
    hops: int,
    reason: str,
)
```

New backend events:

```text
graph_scope       resolved seed entities and candidate count
graph_document    one document added by graph retrieval, with path summary
graph_ablation    plain/identity/graph document counts and graph-only count
conflict_found    competing claim IDs and values
winner_chosen     accepted claim ID, trust score, and rationale
```

### Verification gate

- A read-only HydraDB probe returns one known path using the documented procedure.
- The exact response shape is recorded in a test fixture or planning note.
- No production files are changed in this phase.

### Anti-pattern guards

- Do not invent `document_neighbors`, `get_neighbors`, or path-procedure parameters without implementing or verifying them.
- Do not assume whole nodes are returned from ordinary `MATCH`; the existing client documents HydraDB’s typed result limitations.
- Do not use arrays as HydraDB properties; current property values are scalar.

## 5. Phase 1 — repair correctness before adding graph recall

### Objective

Remove known false identities, false graph edges, false semantic claims, and broken endpoints.

### Required reading

- `src/glasshouse/ask.py`: `_identifier_shaped`, `read_identities`, `connect`, `stream`
- `src/glasshouse/priors.py`: `Priors.is_functional`
- `src/glasshouse/corpus.py`: structured identity extraction
- `src/glasshouse/answer.py`: `SYSTEM`, `write_streaming`
- `src/glasshouse/server.py`: `/api/entity/{eid}`
- all current tests

### Allowed files

- `src/glasshouse/ask.py`
- `src/glasshouse/answer.py`
- `src/glasshouse/server.py`
- new or existing files under `tests/`

### Tasks

1. Make `read_identities` reject functional/role aliases.
   - Apply learned functional-mailbox priors, not only “has an email.”
   - Reject `Security Lead`, `Marketplace Onboarding`, and equivalent role accounts in regression fixtures.
   - Preserve real multi-surface people.
2. Extract one exact document-linking helper.
   - The helper must consume parsed identity fields or token-aware matches.
   - Delete all `surface in document_text.lower()` identity linking.
   - A surface `sam` must not match `same`, `sample`, or an email localpart unless the parser emitted that identity.
3. Change graph semantics in the prompt.
   - `MENTIONED_IN` means co-occurrence only.
   - Remove any instruction that it proves collaboration, ownership, agreement, or co-ownership.
4. Repair streaming-prefix handling.
   - Buffer until `NOT_IN_CORPUS` is confirmed or disproved.
   - If confirmed, discard the entire marker.
   - If disproved, flush every buffered character exactly once.
5. Repair `/api/entity/{eid}`.
   - Resolve public string `eid` to the stored integer `node_id` through the ontology lookup, or query the graph by its string `eid` property.
   - Unknown IDs must return a controlled 404 or an empty documented response, never a HydraDB type error.

### Patterns to copy

- Functional filtering: `Asker.resolve` in `src/glasshouse/ask.py`.
- Exact structured surfaces: `SurfaceIndex.add_document` in `src/glasshouse/resolve.py`.
- Integer graph IDs: `node_id` and `Person.node` in `src/glasshouse/graph.py` and `src/glasshouse/ask.py`.

### Required tests

- real person accepted; role mailbox rejected;
- ambiguous short name rejected;
- `sam` does not link a document containing only `sample`;
- exact parsed `sam` mention does link;
- streaming chunks `['NOT_', 'IN_CORPUS', ' missing']` never expose marker text;
- streaming chunks `['N', 'ormal answer']` produce `Normal answer` exactly;
- entity endpoint supplies an integer node ID and handles unknown EIDs;
- prompt describes a shared document as co-occurrence only.

### Verification commands

```bash
.venv/bin/python -m pytest -q
rg -n " in d\.text\.lower\(\)|co-own|proof of collaboration" src tests
```

### Gate

- All tests pass.
- The grep returns no unsafe substring linker or ownership inference.
- Re-run the saved retrieval sample; identity expansion must not create the known role identities.

## 6. Phase 2 — load the full document–entity graph offline

### Objective

Move HydraDB before retrieval by loading document/entity relationships for the complete normalized corpus.

### Required reading

- `src/glasshouse/corpus.py`: normalized document schema
- `src/glasshouse/resolve.py`: exact surface extraction and kinds
- `scripts/intake.py`: JSONL serialization
- `scripts/load_graph.py`: batching and lookup construction
- `src/glasshouse/graph.py`: deterministic IDs and idempotent writes

### Allowed files

- new `scripts/load_document_graph.py`
- optional new `src/glasshouse/linking.py`
- focused tests under `tests/`
- `README.md` only after the loader works

Do not place this logic in `Asker.connect`. Query-time code sees only already-retrieved documents and cannot build the corpus graph.

### Input schema

Read every `data/normalized/{source}.jsonl` record. Use only structured identity fields:

| Field | Alias kind |
|---|---|
| `emails` | `email` |
| `named_emails[].email` | `email` |
| `named_emails[].name` | `name` |
| `speakers` | `name` |
| `mentions` | `handle` |
| `attendees[].name` | `name` |

Do not scan body text with substring matching.

### Tasks

1. Load the ontology lookup into memory as:

   ```python
   (kind, normalized_surface) -> unique entity record
   ```

   Ambiguous surfaces must map to no entity.
2. Stream normalized JSONL one record at a time; never load a source shard fully into memory.
3. For each document:
   - upsert one `Document` node containing `id`, `doc_id`, `source`, `title`, `date`;
   - collect exact resolved entities from structured fields;
   - deduplicate entities within that document;
   - merge one `Entity-[:MENTIONED_IN]->Document` edge per entity/document pair;
   - store scalar evidence such as `mention_count` and a comma-delimited `kinds` string if supported.
4. Use deterministic node and edge IDs, so reruns are idempotent.
5. Batch at no more than 1,000 rows. The engine rejects batches above 1,024.
6. Add `--source`, `--limit`, and `--offset` for smoke tests.
7. Add a checkpoint in `data/state/` recording completed source and line number. A restart must resume safely.
8. Print counts and rates: documents, linked documents, unique entity/document edges, ambiguous surfaces skipped, unresolved surfaces skipped.

### Patterns to copy

- batching and flush order: nested `flush` in `scripts/load_graph.py`;
- idempotent edges: `GraphEngine.merge_edges`;
- deterministic IDs: `node_id(f"doc:{doc_id}")` and `node_id(f"entity:{eid}")`;
- streaming JSONL: `iter_normalized` in `src/glasshouse/recall.py`.

### Required tests

Use a tiny temporary normalized shard and temporary ontology SQLite file:

- exact email, handle, and name resolve;
- ambiguous alias creates no edge;
- duplicate aliases for one entity create one edge;
- role alias creates no edge if Phase 1 marked it functional;
- short substrings create no edge;
- rerunning the same batch produces identical IDs and no duplicate logical edges;
- checkpoint resumes after the last completed record.

### Verification commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/load_document_graph.py --source slack --limit 100
.venv/bin/python scripts/load_document_graph.py --source slack --limit 100
```

Then query HydraDB and prove:

- document node count is unchanged on the second run;
- edge count is unchanged on the second run;
- one known alias resolves to an entity connected to a real document.

### Gate

- The 100-document smoke test is correct and idempotent.
- The full loader runs or resumes across all 511,962 documents.
- HydraDB contains document/entity edges before any question is asked.

### Anti-pattern guards

- Do not store document bodies in HydraDB.
- Do not create `OWNS`, `DECIDED`, or `COLLABORATES_WITH` from co-occurrence.
- Do not use `CREATE` for replayable edges.
- Do not delete the graph before a normal reload.

## 7. Phase 3 — graph-scoped document retrieval

### Objective

Use HydraDB to add evidence before answer synthesis, including documents absent from plain and alias-expanded FTS results.

### Required reading

- Phase 0 verified API note
- `src/glasshouse/graph.py`
- `src/glasshouse/recall.py`
- `src/glasshouse/ask.py`
- `scripts/score.py`

### Allowed files

- `src/glasshouse/graph.py`
- `src/glasshouse/recall.py`
- `src/glasshouse/ask.py`
- focused tests

### Retrieval algorithm

Implement three explicit strategies; never overwrite one variable and lose the ablation:

```text
plain_docs     = FTS(question)
identity_docs  = FTS(topic AND all aliases of named entities)
graph_scope    = HydraDB documents reachable from named entities
graph_docs     = topic-ranked documents restricted to graph_scope
final_docs     = dedupe(identity_docs + graph_docs), retaining provenance
```

Start with direct entity-to-document neighbors. Add two-entity or multi-hop traversal only after direct graph-scoped retrieval passes evaluation.

### Tasks

1. Add a typed graph-candidate representation containing document ID and path provenance.
2. Add `GraphEngine` queries for:
   - direct documents connected to one or more seed entity node IDs;
   - verified batched paths when a question names two or more entities.
3. Add `LocalRecall.get_many(doc_ids)` preserving requested order.
4. Add topic ranking restricted to a bounded graph candidate set.
   - Use an FTS query plus a temporary scope table or documented bounded batching.
   - Do not fetch thousands of full bodies into Python and implement an ad-hoc scorer.
5. Add `Asker.retrieve(question, limit)` returning a structured result:

   ```python
   RetrievalResult(
       plain_docs,
       identity_docs,
       graph_docs,
       final_docs,
       named_entities,
       graph_candidates,
   )
   ```

6. Make both `stream` and non-streaming callers use this one retrieval method.
7. Stop writing the canonical document graph at query time. `connect` may render/query existing paths but must not redefine linking semantics.
8. Emit `graph_scope`, `graph_document`, and `graph_ablation` events.

### Ranking rules

- Graph reachability is a candidate generator, not proof that the document answers the question.
- Topic relevance still ranks graph candidates.
- Preserve the original FTS rank and graph provenance separately.
- Cap graph expansion and expose the cap as a constant.
- Prefer a direct document connected to the named entity over a collaborator expansion.
- Never label co-occurrence as ownership or a decision.

### Required tests

Create a small fixture where:

- the question names `Sam`;
- plain FTS cannot find a document because it contains neither `Sam` nor the question wording;
- ontology resolves `Sam` to the same entity as `S. Ratnaparkhi`;
- HydraDB links that entity to the answer document;
- graph-scoped topic ranking returns the answer document;
- provenance contains the seed entity and document path;
- unrelated co-mentioned documents do not outrank topic-relevant documents;
- no named entity results in plain FTS only.

### Verification commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/ask.py --trace "What did Sam decide about the retention policy?"
```

### Gate

At least one reproducible real-corpus question satisfies:

```text
expected document not in plain top 20
expected document not in identity top 20, or ranked materially lower
expected document in graph-aware top 20
final answer cites that graph-added document
```

If no real question satisfies this gate, stop. Do not start conflict or UI work. Inspect graph coverage and ranking first.

## 8. Phase 4 — build the A/B evaluation and lock the demo case

### Objective

Prove whether HydraDB improves retrieval, instead of assuming it does.

### Required reading

- `scripts/score.py`
- `data/state/retrieval_score.json`
- blind benchmark schema
- gold-key isolation comments in `scripts/fetch_corpus.py`

### Allowed files

- `scripts/score.py`
- optional new `scripts/find_graph_wins.py`
- tests for scoring logic
- generated ignored reports under `data/state/`

### Tasks

1. Score three strategies independently:
   - `plain`
   - `identity`
   - `graph`
2. Record recall@5, recall@10, and recall@20.
3. Report wins and losses for each transition:
   - identity versus plain;
   - graph versus identity;
   - graph versus plain.
4. Stratify by every benchmark category. Do not report only the first 120 basic questions.
5. Report identity resolution rate, graph coverage rate, median/p95 latency, and number of graph-only retrieved documents.
6. Output machine-readable JSON and a concise terminal table.
7. Add a command that prints the best 5–10 graph-win questions without printing gold answers.
8. Select one stable demo question and one abstention question. Store only question IDs/text and observed public metrics, never gold answers, in a demo fixture.

### Minimum success criteria

- No known false role expansion.
- Graph-aware recall has positive net wins over identity-aware recall.
- At least one non-basic category shows a real graph win.
- The chosen demo case is deterministic across three runs.
- Added latency remains acceptable for a live demo; target p95 under 10 seconds end to end.

### Verification commands

```bash
.venv/bin/python scripts/score.py --all-types --limit 200 --k 20
.venv/bin/python scripts/find_graph_wins.py --limit 10
.venv/bin/python -m pytest -q
```

### Gate

Do not claim graph-native retrieval in README or video until the JSON report shows a positive graph delta and names the evaluated sample size.

## 9. Phase 5 — claims, contradiction, and arbitration

### Objective

Make the graph select between conflicting facts with stored provenance and rationale.

This phase begins only after Phase 4 proves graph retrieval.

### Required reading

- local design spec ontology and trust sections
- `src/glasshouse/answer.py`
- `src/glasshouse/config.py`
- `src/glasshouse/graph.py`
- Ollama client usage already present in `src/glasshouse/answer.py`

### Allowed files

- new `src/glasshouse/claims.py`
- new `src/glasshouse/trust.py`
- `src/glasshouse/graph.py`
- `src/glasshouse/ask.py`
- `src/glasshouse/answer.py`
- focused tests

### Narrow first implementation

Support a small explicit predicate vocabulary first:

```text
owner
status
due_date
limit
reports_to
```

Unknown predicates remain evidence text and are not force-normalized.

### Claim model

```python
Claim(
    claim_id: str,
    subject_eid: str,
    predicate: str,
    object_value: str,
    object_eid: str | None,
    doc_id: str,
    source: str,
    asserted_at: str | None,
    author_eid: str | None,
    extractor_confidence: float,
    trust: float,
    status: str,  # accepted | disputed | superseded
    rationale: str,
)
```

HydraDB representation:

```text
(:Claim)-[:SUBJECT]->(:Entity)
(:Claim)-[:OBJECT]->(:Entity)          # only when object_eid exists
(:Claim)-[:EVIDENCED_BY]->(:Document)
(:Claim)-[:CONTRADICTS]->(:Claim)
(:Claim)-[:SUPERSEDES]->(:Claim)
```

### Tasks

1. Extract claims only from the final candidate documents, not the whole corpus.
2. Cache extraction by document content hash and extractor version.
3. Parse model output defensively:
   - temperature 0;
   - prompt includes exact JSON schema;
   - validate every field;
   - one repair attempt;
   - invalid output becomes no claim, never guessed data.
4. Group conflicts by `(subject_eid, predicate)` and differing normalized objects.
5. Compute deterministic trust from source authority, predicate-sensitive recency, corroboration, author role when known, explicitness, and extractor confidence.
6. Store both winning and losing claims.
7. Store `CONTRADICTS`, `SUPERSEDES`, `status`, trust, and rationale.
8. Supply answer synthesis with structured claim records, not vague path summaries.
9. Emit `conflict_found` and `winner_chosen` events.
10. Remove or soften every product claim about arbitration until this path passes tests.

### Required tests

- same subject/predicate/same object is corroboration, not conflict;
- same subject/predicate/different object is conflict;
- newer volatile claim can supersede older claim;
- stable predicate can prefer authoritative/corroborated evidence;
- losing claim remains queryable;
- rationale names only signals actually used;
- malformed model JSON creates no claim;
- answer cites the winning document and acknowledges material disagreement;
- low trust triggers confidence abstention.

### Gate

- Run all available `conflicting_info` questions.
- Record conflict retrieval coverage and answer correctness separately.
- Lock one deterministic conflict demo where the UI shows both claims, the winner, and why.

## 10. Phase 6 — product and demo integration

### Objective

Show the two moments where HydraDB changes the result: graph retrieval and conflict arbitration.

### Allowed files

- `src/glasshouse/web/index.html`
- `src/glasshouse/server.py`
- static landing files under `web/`
- UI-focused tests if present

### Tasks

1. Add a visible A/B panel:
   - plain search documents;
   - graph-added documents;
   - the path that added each graph document.
2. Label `MENTIONED_IN` paths as co-occurrence, never ownership.
3. Add conflict presentation:
   - both values;
   - source and date for each;
   - accepted value;
   - trust rationale.
4. Keep the answer citations clickable/inspectable.
5. Add an explicit abstention state for linking, connectivity, and confidence gates.
6. Make the selected graph-win demo complete in under 60 seconds of video time.

### Three-minute video outline

```text
0:00–0:20  Nine tools disagree; search does not know identity or truth.
0:20–1:05  A/B graph-retrieval question: plain misses, HydraDB path finds evidence.
1:05–1:50  Conflict question: both claims survive, winner and rationale shown.
1:50–2:20  Abstention question: system refuses with an explicit gate.
2:20–2:45  Architecture: Alias/Entity/Document/Claim graph and HydraDB traversal.
2:45–3:00  Measured results and repository/deployment link.
```

### Gate

- The browser demo works from a fresh server start.
- No UI claim exceeds backend behavior.
- The video rehearsal is under 2:50, leaving ten seconds of safety.

## 11. Phase 7 — cold-clone verification and submission

### Objective

Make the public repository understandable and runnable by judges.

### Required fixes

The current README setup stops too early. It must include:

```text
fetch_corpus.py
intake.py
build_index.py
resolve_entities.py
load_graph.py
load_document_graph.py
server start command
evaluation command
```

Also document expected runtime, disk use, keys, Docker requirements, generated files, and a smaller `--limit` smoke path.

### Verification checklist

- all local commits are pushed to the public repository;
- repository is public;
- license and third-party attribution remain present;
- fresh environment installs `.[dev]`;
- `docker compose up -d` starts HydraDB;
- 100-document smoke setup completes from documented commands;
- all tests pass;
- deployed/local demo link opens in a private browser window;
- video link opens without access request;
- submission form contains repository, video, description, HydraDB explanation, stack, and contribution details.

### Final anti-pattern grep

```bash
rg -n "co-own|proof of collaboration|active winner|authority priors" README.md web src
rg -n "surface in| in d\.text\.lower\(\)" src scripts
git status --short
git log --format='%h %ad %s' --date=iso-strict
```

Every marketing statement found by the first grep must have a tested implementation or be rewritten.

## 12. Work division

Use consecutive agents, not simultaneous edits to shared core files.

| Work package | Owner | Files | Depends on |
|---|---|---|---|
| A. Correctness fixes | implementation agent A | `ask.py`, `answer.py`, `server.py`, tests | Phase 0 |
| B. Offline graph loader | implementation agent B | new loader/linking module, tests | A |
| C. Graph query API | implementation agent C | `graph.py`, `recall.py`, tests | B |
| D. Query integration | implementation agent D | `ask.py`, event tests | C |
| E. A/B evaluation | implementation agent E | scoring scripts, tests | D |
| F. Claims/conflicts | implementation agent F | claims/trust modules, `graph.py`, `ask.py`, tests | E gate passes |
| G. UI/demo | implementation agent G | server/web/static files | D and F |
| H. Docs/submission | implementation agent H | README and submission artifacts | all verified |

After each work package, use separate agents for:

1. verification commands;
2. anti-pattern grep;
3. code-quality review;
4. commit only after all three pass.

## 13. Cut line

If time becomes tight, cut in this order:

1. ontology browse screen;
2. hosted deployment; record localhost if necessary;
3. broad predicate support; keep only predicates needed by the conflict demo;
4. collaborator/multi-hop expansion beyond direct entity-to-document retrieval;
5. evaluation breadth, while retaining every conflict and not-found question possible.

Never cut:

- exact identity linking;
- one measured graph-retrieval win;
- one real conflict with both sources and rationale;
- one honest abstention;
- reproducible tests;
- public repository, license, setup instructions, video, and submission form.

## 14. Definition of done

Glasshouse is complete enough to submit when all of the following are true:

- HydraDB contains the document/entity graph before query time.
- At least one real benchmark answer document is retrieved only through the graph-aware strategy.
- Evaluation reports a positive graph delta rather than only an anecdote.
- A conflict query stores and displays both claims and an auditable winner.
- An unanswerable query abstains through a named gate.
- The live UI shows graph-added evidence and claim provenance.
- All tests pass from the documented environment.
- The public repository contains the current implementation and complete setup instructions.
- The demo video is under three minutes and all submission links work.
