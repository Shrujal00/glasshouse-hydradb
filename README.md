<div align="center">

<img src="docs/assets/hero.png" alt="Scattered documents converging into a resolved graph, with two conflicting nodes lit in orange" width="100%">

# Glasshouse

**An enterprise ontology you can see through.**

Nine enterprise tools disagree about who owns what. Glasshouse resolves the
identities, arbitrates the contradictions, admits when the answer isn't there —
and shows its work while it does.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Built on HydraDB](https://img.shields.io/badge/built%20on-HydraDB-ff5c39.svg)](https://hydradb.com)
[![Hack Hydra](https://img.shields.io/badge/Hack%20Hydra-Track%2001-6e56cf.svg)](https://hackhydra.hydradb.com)

</div>

---

> **511,962 documents. Nine sources. One graph.** Ask a question in English and
> watch the reasoning happen: identities resolving, the graph opening, sources
> disagreeing, and — where the evidence does not support a verdict — the system
> declining to give one.

---

## Contents

- [What Glasshouse does](#what-glasshouse-does)
- [Core capabilities](#core-capabilities)
- [How HydraDB is used](#how-hydradb-is-used)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [The pipeline, end to end](#the-pipeline-end-to-end)
- [Modules in depth](#modules-in-depth)
- [Designing for the engine](#designing-for-the-engine)
- [Results](#results)
- [Attribution](#attribution)

---

## What Glasshouse does

A company keeps the same fact in nine different tools. Over time those facts
drift apart, and nobody notices — because nobody is looking. You find out when
somebody acts on the wrong one.

Glasshouse does three things about that:

1. **Resolves who everyone is.** 401,163 written forms across nine tools
   collapse into 166,429 people — and 163,262 proposed merges are *refused*,
   because a hard rule said those two are not the same person.
2. **Pulls competing statements out as records.** Typed claims with a subject,
   a value, a source and a date, extracted only from text the answer itself was
   shown.
3. **Decides between them in code, or refuses.** Source authority, recency,
   corroboration, hedging. No model call — the same claims arbitrate the same
   way every time, and the reason shown to the reader is the reason the answer
   came out that way.

The result is a map of **what the organisation contradicts itself about**,
built without anybody asking a question.

---

## Core capabilities

| Capability | Where |
|---|---|
| Parsing nine tool formats into one shape | `corpus.py`, `scripts/intake.py` |
| BM25 + facet index over 511,962 documents | `recall.py`, `scripts/build_index.py` |
| Entity resolution — blocking, pair scoring, hard constraints | `resolve.py` |
| Identity as a graph traversal (`Surface -[:DENOTES]-> Entity`) | `graph.py` |
| Three entrances into the graph: person, evidence, container | `ask.py` |
| Claim extraction over a fixed predicate vocabulary | `claims.py` |
| Deterministic arbitration with abstention | `trust.py` |
| Contradiction stored as edges, found offline | `scripts/load_claims_graph.py` |
| Disagreement map, supersession chains, blast radius | `graph.py`, `server.py` |
| Managed-cloud hybrid retrieval (dense + sparse) | `cloud.py` |
| Streaming UI — disagreements, ontology, ask | `server.py`, `web/index.html` |
| Retrieval scoring and per-fact answer grading | `scripts/score.py`, `scripts/grade.py` |

183 tests, no network required to run them.

---

## How HydraDB is used

HydraDB is used **twice, for two different jobs**, and the product does not
work without either.

### The open-source engine — the ontology and the reasoning

Self-hosted in Docker. Holds who everybody is and what contradicts what:

```
(:Surface {text, kinds, entities})-[:DENOTES]->(:Entity)          209,388
(:Alias {surface, kind})-[:RESOLVES_TO {score, signals}]->(:Entity)
(:Entity {eid, canonical_name})-[:MENTIONED_IN|SPOKE_IN|SENT]->(:Document)
(:Document)-[:IN_CONTAINER]->(:Container {key, source, kind})      978,512 edges
(:Claim {predicate, subject, object_value, asserted_at, trust, status})
(:Claim)-[:EVIDENCED_BY]->(:Document)
(:Claim)-[:ABOUT]->(:Entity)
(:Claim)-[:CONTRADICTS]->(:Claim)
(:Claim)-[:SUPERSEDES]->(:Claim)
(:Disagreement)-[:OVER]->(:Claim)
```

Four questions become one traversal each:

| Question | Traversal | Measured |
|---|---|---|
| Who is `@jae`? | one anchored `DENOTES` hop | ~0.09s |
| Which page is in this space? | one hop into `Container` | 159,030 documents → 6 |
| What do we contradict ourselves about? | rank `Disagreement` nodes | 0.06s |
| Who read the version that was wrong? | `Claim → Document ← Entity` | 3 hops |

**Turn the engine off and all four return zero.** It is not storing a result
computed elsewhere; it is where the reasoning lives.

### The managed cloud — recall

`cloud.py` puts documents into HydraDB Cloud and asks it which handful are
worth reading. Its hybrid retrieval brings **dense embeddings this stack could
not otherwise have** — every Ollama embedding model returns 401 on this account
and the local GPU is 6 GB — so a question that paraphrases (`"too many requests
errors"` for `429`) can still find its document.

---

## Tech stack

- **Python ≥ 3.11**, no framework — `pyproject.toml` for dependencies
- **HydraDB open-source engine** in Docker — the ontology and contradiction graph
- **HydraDB Cloud** (`hydra-db` SDK) — hybrid document recall
- **SQLite FTS5** — BM25 over 511,962 documents, and the facet store
- **FastAPI + server-sent events** — streaming so the reasoning is watchable
- **Vanilla HTML/CSS/JS** — one file, no build step, force-directed canvas
- **Ollama Cloud** (`gpt-oss:120b`) — claim extraction and answer synthesis only
- **pytest** — 183 tests

---

## Repository structure

```
src/glasshouse/
  corpus.py      parse nine tool formats into one document shape
  recall.py      SQLite FTS5 + BM25, facet-weighted
  facets.py      folders, channels, spaces, speakers — the container store
  resolve.py     entity resolution: blocking, scoring, hard constraints
  graph.py       HydraDB open-source engine: traversals and the read API
  cloud.py       HydraDB Cloud: ingest and hybrid recall
  claims.py      claim extraction over a fixed predicate vocabulary
  trust.py       deterministic arbitration and abstention
  ask.py         the three entrances, retrieval, and the event stream
  answer.py      passage selection and synthesis
  priors.py      per-source and per-role priors
  server.py      FastAPI: the three screens and their endpoints
  web/index.html the interface — disagreements, ontology, ask

scripts/
  fetch_corpus.py        pull the benchmark corpus
  intake.py              → data/normalized/*.jsonl
  build_index.py         → the BM25 index
  build_facets.py        → the facet store
  resolve_entities.py    → entities.jsonl
  load_graph.py          entities and aliases → HydraDB
  load_document_graph.py documents and mention edges
  load_facet_graph.py    containers, speakers, senders
  load_surface_graph.py  Surface nodes — the query-time entrance
  load_claims_graph.py   the contradiction graph
  ingest_cloud.py        documents → HydraDB Cloud
  score.py               retrieval scoring
  grade.py               per-fact answer grading
```

---

## Getting started

### Prerequisites

- Python 3.11+
- Docker (for the HydraDB open-source engine)
- An Ollama Cloud key (claim extraction and synthesis)
- A HydraDB Cloud key (optional — only for `cloud.py`)

### Install

```bash
git clone https://github.com/Shrujal00/glasshouse-hydradb.git
cd glasshouse-hydradb

python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env        # then fill it in
docker compose up -d        # the HydraDB open-source engine
.venv/bin/python -m pytest  # 183 tests, no network needed
```

### Environment variables

| Variable | Required | What for |
|---|---|---|
| `HYDRA_LOCAL_TOKEN` | yes | auth for the local engine — 32+ characters |
| `HYDRA_HTTP_URI` | yes | defaults to `http://127.0.0.1:8443` |
| `OLLAMA_API_KEY` | yes | claim extraction and answer synthesis |
| `OLLAMA_HOST` | no | defaults to `https://ollama.com` |
| `HYDRA_DB_API_KEY` | no | HydraDB Cloud recall |
| `GOLD_ANSWERS_PATH` | no | grading only — **outside the repo**, and nothing under `src/glasshouse` may read it |

### Build the indexes and the graph

Each step is restartable and prints what it did. Timings are measured on the
full corpus.

```bash
.venv/bin/python scripts/fetch_corpus.py
.venv/bin/python scripts/intake.py               # → data/normalized/*.jsonl

# local indexes — no network, no key
.venv/bin/python scripts/build_index.py          # FTS5 + facets  4.3 GB  ~3.5 min
.venv/bin/python scripts/build_facets.py         # facet store    0.4 GB  ~25s

# the ontology: 401,163 written forms → 166,429 identities
.venv/bin/python scripts/resolve_entities.py     # 22.6s
.venv/bin/python scripts/load_graph.py           # entities + aliases  ~3 min

# HydraDB: documents before containers, containers before surfaces
.venv/bin/python scripts/load_document_graph.py
.venv/bin/python scripts/load_facet_graph.py     # 1.23M edges  ~20 min
.venv/bin/python scripts/load_surface_graph.py   # 209,388 surfaces

# the contradiction graph — extraction is a model call, so this one costs
.venv/bin/python scripts/load_claims_graph.py --keys 260 --dry-run
.venv/bin/python scripts/load_claims_graph.py --keys 260
```

**Order matters.** Every loader after `load_graph.py` matches endpoints that
must already exist; run one early and it writes edges that connect to nothing.

`load_claims_graph.py` checkpoints every work item, so an interrupted run
resumes without re-paying for the same extraction. If a whole run comes back
with no claims, check for a `429` before believing the corpus is quiet — a
rate-limited extraction returns valid JSON asserting nothing.

### Run it

```bash
.venv/bin/python -m uvicorn glasshouse.server:app \
    --host 127.0.0.1 --port 8080 --app-dir src
```

Open `http://127.0.0.1:8080`. It lands on the disagreement map.

---

## The pipeline, end to end

```
nine tool exports
      │  scripts/intake.py — one document shape
      ▼
511,962 normalized documents
      │  scripts/build_index.py — BM25 + facets
      │  scripts/ingest_cloud.py — HydraDB Cloud recall
      ▼
      │  scripts/resolve_entities.py — 8.7M pairs scored,
      │     42,959 merges accepted, 163,262 refused
      ▼
166,429 identities ──► scripts/load_graph.py + load_surface_graph.py
      │                    (:Surface)-[:DENOTES]->(:Entity)
      ▼
      │  scripts/load_document_graph.py + load_facet_graph.py
      │     mentions, speakers, senders, containers
      ▼
      │  scripts/load_claims_graph.py
      │     claims.py extracts → trust.py arbitrates → edges
      ▼
(:Claim)-[:CONTRADICTS]->(:Claim)   the disagreement map
```

At query time `ask.py` opens whichever entrance the question allows — a person,
the evidence, or a container — arbitrates whatever claims come back, and
streams every step to the interface as it happens.

---

## Modules in depth

### `corpus.py` — parsing
Nine formats into one shape. `derive_date` recovers dates from document
*bodies* — activity logs, revision histories — because seven of nine sources
set a `date` field on under 2% of their documents. Coverage went **26% → 51%**,
and 0% → ~97% on the ticket and CRM documents where `status` and `owner` claims
live.

### `resolve.py` — the ontology
Blocking, pairwise scoring, union-find clustering, and hard constraints. Two
constraints do most of the precision work: one person has one corporate
mailbox, and shared mailboxes are excluded outright. **163,262 merges were
refused** — nearly four times more than were accepted. That ratio is the
difference between an ontology and string similarity.

### `graph.py` — the engine
Every read is anchored on a literal node id, because `node_id()` is a
deterministic int64 digest of a string key and nothing else in this engine is
addressable. Holds the traversals, the contradiction read API, and the
measured constraints that shape both.

### `claims.py` — extraction
Five predicates: `owner`, `status`, `due_date`, `limit`, `reports_to`. Anything
outside them is dropped rather than force-fitted. Every value is checked
against the text the model was actually shown — a value that is not there is
discarded, which is what stops a plausible-sounding colleague's name reaching
the reader as fact.

### `trust.py` — arbitration
Arithmetic, not a model call: source authority, predicate-sensitive recency,
corroboration, hedging, extraction confidence. Ten runs with the input order
shuffled produce byte-identical verdicts. **It is allowed to refuse** — and
where no individual signal can be named, it does, because a verdict whose
rationale reads "no signal separated these claims" is a verdict contradicting
itself.

### `ask.py` — the three entrances
Evidence-first, identity-expanded, and container. Every question runs through
all three at once and the trace records which one reached the document, so the
graph has to earn its place on each answer. The container hop is the one that
finds pages keyword search cannot: it walks to the folder the question named
and returns only what is inside it.

---

## Designing for the engine

Every pattern below was probed against the running engine. They are the reason
the architecture looks the way it does, and they are what makes the graph fast.

| Pattern | Measured |
|---|---|
| `MATCH (s:Surface {id: …})-[:DENOTES]->(e:Entity)` — identity | **~0.09s** |
| `MATCH (d:Disagreement) … ORDER BY … LIMIT` — the map | **0.06s** |
| container hop, 159,030 documents scoped to the folder | **6 documents** |
| a scan with `WHERE`, `ORDER BY` and relationships over a small label | **0.03s** |
| `algo.SSpaths`, maxLen 3 | **1.0s** |

Three rules follow, and the whole design is built on them:

**Anchor every read on a literal node id.** Addressing is deterministic —
`node_id(key)` is a blake2b digest — so any node in the graph can be reached
directly without searching for it.

**Make one read, not many.** Reads are anchored and individual, so the loader
bounds how many a single answer needs.

**Precompute at load time what a scan would otherwise compute.** A `Surface`
node carries how many people its form could mean, so the question is answered
by reading a property rather than counting.

The first rule is about label *size*, not query shape. Over a few hundred nodes
a full scan with `WHERE`, `ORDER BY` and relationships runs in 0.03s — which is
exactly why the disagreement map is a real graph query rather than a lookup
table, and why keeping that label small is the loader's job.

Loads are generational: every load stamps its nodes with a generation and the
reader asks for one stamp, so the map always reflects a single coherent build.

---

## Results

Measured with `scripts/grade.py`, which grades every answer against the
benchmark's own `answer_facts` rubric with an LLM judge, one fact at a time.
The answer key lives **outside this repository** and only the two scoring
scripts may read it; nothing under `src/glasshouse` can see it.

Every number below is 20 questions of that type, graded fact by fact. The
current run is 2026-08-21 and its raw report is committed at
`data/state/answer_grade.json`, so it can be checked without re-running
anything.

| Category | Answers | Fact recall |
|---|---:|---:|
| `info_not_found` — knowing the answer is absent | 20 | **100%** |
| `intra_document_reasoning` | 20 | **82%** |
| `basic` | 20 | 41% |
| `project_related` — up to 20 required facts each | 20 | 38% |
| `semantic` — deliberate paraphrase | 20 | 35% |
| **overall** | **100** | **199 / 424 — 46.9%** |

43 of those 100 answers supported every required fact. Median latency 10.1s.

An earlier run of the same harness on 2026-08-19 measured three types this run
did not reach — `constrained` **67%**, `conflicting_info` **59%**,
`completeness` **32%** (10 answers). They are reported separately because they
are a different run, against `data/state/grade_final.log`.

The metadata work below was measured at the retrieval layer with
`scripts/score.py` rather than with the answer judge, and is quoted as such.

What moved, and why:

- **`metadata` retrieval doubled, 8/30 to 16/30 expected documents.** The facet
  fields were normalized and then discarded, so the answer to "which space is
  this page in" was never in the index. Indexing the facets and adding the
  container hop put it there. This is a retrieval measurement rather than a
  graded answer score.
- **`conflicting_info` went from ~52% to 59%**, after claims arbitration and
  after recovering the dates that arbitration ranks on.
- **`info_not_found` held at 100%** through every prompt change — including the
  ones that deliberately made the system more willing to answer elsewhere. That
  it did not move is the result.

One change mattered more than any retrieval work: letting the answer path speak
whenever the evidence is in the context, rather than gating on whether a person
resolved first. "Who authored the SLO throttler PR" names no person and its
answer *is* a person — the document was already there, and now it is answered.

<!-- RESULTS -->

## Attribution

Built by **[Shrujal Ganatra](https://github.com/Shrujal00)** for Hack Hydra 2026.
[LinkedIn](https://www.linkedin.com/in/shrujal-ganatra/)

Licensed under **Apache-2.0**. You are free to clone, fork and build on this — the
[`NOTICE`](NOTICE) file must travel with any derivative, and the licence grants no
rights to use the author's name or branding to endorse your fork.

**Third-party:**

- [HydraDB](https://github.com/hydra-db/hydradb) — AGPL-3.0, run as a separate
  containerised service, not linked into this codebase
- [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) by
  Onyx — not redistributed here, fetched by script
- [Salesforce HERB](https://huggingface.co/datasets/Salesforce/HERB) —
  CC-BY-NC-4.0, research and non-commercial use only, not redistributed
- [Ollama](https://ollama.com)

<div align="center">
<sub>Built in public, 12–20 August 2026.</sub>
</div>
