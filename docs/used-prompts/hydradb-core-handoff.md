# Glasshouse Handoff — make HydraDB undeniable

Continue building Glasshouse (HackHydra **Track 01: enterprise context and
ontology**). Deadline **21 August 2026, 18:29 IST**.

**The goal is not the benchmark.** It is that a judge looks at this and thinks
*"I did not know you could do that with a graph."* Everything below serves that.

Read `README.md` first, then this. Where they disagree, this wins.

## State

Two branches, both working, both pushed.

- **`main`** — 154 tests. Benchmark ≈44% weighted fact recall (from 35.5%).
  Orange UI with an inspectable graph. README with architecture, build
  sequence and negative results. `docs/demo-script.md`.
- **`hydradb-core`** — branched off `main`. Identity resolution moved out of
  SQLite and into HydraDB. **Work here.**

```bash
docker compose up -d
.venv/bin/python -m uvicorn glasshouse.server:app --host 127.0.0.1 --port 8080 --app-dir src
.venv/bin/python -m pytest -q          # 154 pass
```

## What the engine can and cannot do — measured, do not re-derive

This is the single most useful thing in this document. Every line was probed
against the running engine, and it is what shapes the architecture.

| Query | Result |
|---|---|
| `MATCH (n:Container) RETURN count(*)` | works — 57,766 |
| `MATCH (n:Entity) RETURN count(*)` | **timeout** (166,429 nodes) |
| `MATCH (n:Document) RETURN count(*)` | **429 resource_exhausted** |
| `MATCH (n:Entity) RETURN count(n) AS n` | rejected — aliased aggregate |
| `MATCH (n) WHERE id(n) = $i RETURN n` | rejected |
| `MATCH (s:Surface {id: 123})-[:DENOTES]->(e:Entity) RETURN e.eid` | **works, ~0.09s** |
| the same statement under `UNWIND` | rejected — "batch node patterns do not support relationships" |
| `CALL algo.SSpaths({sourceNode: <int>, ...}) YIELD path` | works, 0.04–0.27s |
| `CALL algo.SPpaths({sourceNode, targetNode, ...})` | works — shortest path |
| `UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Label, ...` | writes, ~60–335k items/min |

Three rules follow, and they are not negotiable:

1. **Anchor every read on a literal node id.** `node_id("some:key")` is a
   deterministic int64 digest, so any string is an O(1) anchor. Nothing else is
   addressable.
2. **There is no batched read.** Every anchored lookup is one round trip at
   ~0.09s. Bound how many you make; do not assume you can loop.
3. **Anything a scan would compute must be precomputed at load time and stored
   as a node property.** That is why `Surface` carries `entities`,
   `given_name_forms` and `kinds`.

Also: `strong=True` is required to read your own writes. Writes are keyed on a
payload digest for idempotency — see the comment in `GraphEngine.query`.

## What is in the graph now

```
(:Surface {text, kinds, entities, given_name_forms})-[:DENOTES]->(:Entity)
(:Alias {surface, kind})-[:RESOLVES_TO {score, signals}]->(:Entity)
(:Entity {eid, canonical_name, alias_count})-[:MENTIONED_IN]->(:Document)
(:Document)-[:IN_CONTAINER]->(:Container {key, source, kind, name, documents})
(:Entity)-[:SPOKE_IN]->(:Document)
(:Entity)-[:SENT]->(:Document)
```

209,388 surfaces · 166,429 entities · 511,962 documents · 57,766 containers ·
978,512 `IN_CONTAINER` · 159,374 `SENT` · 33,775 `SPOKE_IN`.

Loaders: `load_graph.py` → `load_document_graph.py` → `load_facet_graph.py` →
`load_surface_graph.py`. Order matters; the later ones match existing endpoints
and silently write nothing if the endpoints are absent.

## The job: build the contradiction graph

This is the jaw-drop, and it is the only thing on this list that matters.

Today, when two documents disagree, `claims.py` extracts the competing
statements and `trust.py` scores them — and then throws the reasoning away into
a SQLite cache. **Put it in the graph instead:**

```
(:Claim {id, predicate, subject, object_value, asserted_at, trust, status})
(:Claim)-[:ABOUT]->(:Entity)
(:Claim)-[:EVIDENCED_BY]->(:Document)
(:Claim)-[:CONTRADICTS]->(:Claim)
(:Claim)-[:SUPERSEDES]->(:Claim)
```

`IMPLEMENTATION_PLAN.md` §9 already specifies this and it was never built.

Once contradictions are **edges**, things become possible that no vector store
and no relational schema can do without a bespoke pipeline:

- **A disagreement map of the company.** "What does this organisation
  contradict itself about?" — traverse `CONTRADICTS`, rank by trust gap and by
  how many documents hang off each side. Nobody ships this. It is one query
  over a graph and it is genuinely useful to a real company.
- **Supersession chains.** "What was the burst-credit limit in March, and what
  corrected it?" — walk `SUPERSEDES` backwards and you get the history of a
  fact, with the document that changed it at every step.
- **Blast radius.** Anchor on a `Claim`, hop to `Document`, hop to `Entity`:
  *who has been reading the version that turned out to be wrong.* That is a
  three-hop traversal and a genuinely alarming thing to be able to show.

Build it offline into the graph (a `scripts/load_claims_graph.py` over the
`conflicting_info` documents at minimum — do not try to extract claims for all
511,962), then add one screen that shows the map. The demo is: *here is
everything your company disagrees with itself about, and here is why we believe
one side over the other — and here is where we refuse to choose.*

## Rules — these have bitten before

- **Gold answers** live at `$GOLD_ANSWERS_PATH`, outside the repo. Only
  `scripts/score.py` and `scripts/grade.py` may read them. Nothing under
  `src/glasshouse` may import or see them.
- Commits: **subject line only, no body, never mention Claude or AI.**
- **The trace must not overstate.** Three labels have already been corrected
  for claiming more than the measurement supports. If the local facet table
  served a scope rather than the graph, the trace says so.
- **Do not claim a graph-native *retrieval* win.** Measurement does not support
  it. The graph earns its place on identity and arbitration — say that.
- The user tests the UI themselves. Hand over the URL; do not drive Chrome.
- Ollama Cloud session limits have been hit twice. A rate-limited run records
  every answer as **0 facts** — if a whole category scores exactly zero, check
  for `429` before believing it.

## Known-good demo questions

- `INC-9821: was the degraded GPU node an OOM or intermittent driver/kernel launch stalls?`
  — 16 claims, 3 conflicts, refuses to decide two of them.
- `In the internal customer success and support knowledge space, which published page contains escalation templates?`
  — trace shows `hydradb reached from 1 container`.
- `In the customer success shared drive, which draft spreadsheet owned by Jordan Reyes was last modified most recently?`
  — resolves Jordan Reyes through `DENOTES`, then both document entrances.

## What is deliberately not done, and why

- **The FTS index stays in SQLite.** BM25 over 511,962 documents is the exact
  unanchored scan the engine rejects. Moving it would make the product slower
  and worse. This is a measured decision, not an omission — say so plainly if
  asked rather than apologising for it.
- **Dense retrieval is unavailable.** Every Ollama embedding model returns 401
  on this account, there is no local torch, and the GPU is 6 GB. `semantic`
  (125 questions, 29%) is capped because of it.
- **LLM query expansion made retrieval worse** — 9/30 → 5/30. The model invents
  plausible identifiers like `safest_numeric_mode` that exist in no document.
  Do not retry it.
