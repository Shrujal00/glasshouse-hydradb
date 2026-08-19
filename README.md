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

## What it does

Give a search engine half a million documents from Slack, Gmail, Linear, Google
Drive, HubSpot, Fireflies, GitHub, Jira and Confluence, and it will happily find
you a document. That is not the hard part.

The hard part is what the Hack Hydra Track 01 brief says out loud:

> *"Extraction is the easy part now that LLMs are cheap. The hard part is entity
> resolution and ontology alignment: deciding that "Sam", "@soham" and
> "S. Ratnaparkhi" are one person, and figuring out which of two contradictory
> statements to trust."*

Glasshouse is built around four ideas:

| | |
|---|---|
| **One node per real thing** | `@priya`, `priya_nair@redwood.com` and `Priya Nair` are one person. |
| **Every fact carries its source** | Not *"Sam owns billing"* but *"Sam owns billing — per this page, dated March 3."* |
| **Disagreements are kept, not hidden** | When two documents conflict, both survive. One wins, and the reason is shown. |
| **Knowing when to shut up** | If the answer is not in the corpus, say so — while still surfacing what *is* known. |

Built on **HydraDB**, using both the managed cloud API and the open-source
engine, each for a different half of the problem.

---

## How a question is answered

```
question
   │
   ├─ recall      511,962 documents → a 500-document page          (FTS5, ~60ms)
   ├─ rerank      rescored against what each document records      (~40ms)
   ├─ entrances   three ways into HydraDB, run independently
   ├─ resolve     surface forms → people, via the prebuilt ontology (~1ms)
   ├─ claims      explicit assertions extracted from the evidence
   ├─ arbitrate   competing values scored, or deliberately not
   └─ answer      cited, or an honest account of what is missing
```

### Three entrances into the graph

Most graph-RAG systems have one: find the people named in the question, walk out
from them. That entrance opens for **21 of 570** benchmark questions here — we
measured it. So there are three.

| Entrance | Traversal | Opens when |
|---|---|---|
| **Forward** | `(:Entity)-[:MENTIONED_IN]->(:Document)` | the question names a person |
| **Reverse** | the same edge read inwards | always — gives the people attached to whatever retrieval found |
| **Container** | `(:Document)-[:IN_CONTAINER]->(:Container)` | the question names a *place*: a channel, folder or space |

The reverse entrance exists because *"who owns the audit-log shipper"* names
nobody — the person is the answer, not the query. The container entrance exists
because *"in the internal customer success and support knowledge space"* names
neither a person nor a phrase the document body repeats; it names a **place**,
and places are nodes.

All three are anchored, single-hop `algo.SSpaths` calls. A labelled `MATCH`
expansion over `MENTIONED_IN` scans the whole edge set — 15–30s per seed, often
past the engine's 30s cap. `SSpaths` returns 200 paths in 0.04–0.27s.

### What is in HydraDB

```
(:Surface {text, kinds, entities, given_name_forms})-[:DENOTES]->(:Entity)
(:Alias    {surface, kind})-[:RESOLVES_TO {score, signals}]->(:Entity)
(:Entity   {eid, name})-[:MENTIONED_IN]->(:Document {doc_id, title})
(:Document)-[:IN_CONTAINER]->(:Container {key, source, kind, name, documents})
(:Entity)-[:SPOKE_IN]->(:Document)     -- who talked, not who was mentioned
(:Entity)-[:SENT]->(:Document)         -- mail authorship

(:Claim {predicate, subject, object_value, asserted_at, trust, status})
(:Claim)-[:EVIDENCED_BY]->(:Document)
(:Claim)-[:ABOUT]->(:Entity)
(:Claim)-[:CONTRADICTS]->(:Claim)      -- both directions; neither side is a dead end
(:Claim)-[:SUPERSEDES]->(:Claim)       -- only where recency is what settled it
(:Disagreement {subject, predicate, sides, sources, trust_gap, decided})-[:OVER]->(:Claim)
```

### Identity resolution is a traversal

The brief calls entity resolution the hard part, so it runs in the graph rather
than beside it. A word taken out of a question anchors a `Surface` node by the
deterministic digest of its text, and one `DENOTES` hop reaches every person
that written form could mean:

```
MATCH (s:Surface {id: <digest of "jordan reyes">})-[:DENOTES]->(e:Entity)
RETURN s.entities, s.kinds, e.eid, e.canonical_name, e.alias_count
```

**How many people it reaches is the ambiguity guard.** A form denoting two
people has named neither, and refusing to expand it is the same discipline as
refusing to answer from evidence that is not there.

The engine shapes this design rather than merely tolerating it. It rejects
unanchored scans — `MATCH (n:Entity) RETURN count(*)` over 166,429 entities
times out, and over `Document` returns `429` — and it will not accept the
traversal under `UNWIND`, so there is no batched form. Everything a scan would
have computed is therefore counted once at load time and carried on the node:
`entities` for ambiguity, `given_name_forms` for whether a bare capitalised
word is a name, `kinds` for whether one spelling was used as both a handle and
a name. Query time reads only the node it anchored on.

The cost is real and worth stating: resolution went from a microsecond SQLite
lookup to ~0.09s per word with no batching. Retrieval is ~1.9s cold and ~0.5s
once the per-process memo is warm, and the number of surfaces resolved out of
the retrieved documents is capped at 48 because the canvas draws eight people
and resolving three hundred to show eight was waste.

978,512 `IN_CONTAINER` edges, 159,374 `SENT`, 33,775 `SPOKE_IN`, across 57,765
containers — loaded at ~61,000 items/min.

### Arbitration

Claims are extracted over a fixed predicate vocabulary — `owner`, `status`,
`due_date`, `limit`, `reports_to` — from the *same passages the answer prompt
sees*, so arbitration can never hand the model a value it cannot find in its own
evidence. Competing values are then scored on source authority, recency,
explicitness and corroboration.

Recency is one of those signals, and it nearly didn't work. Only Gmail and
Fireflies record a `date` field; the other seven sources set one on under 2% of
their documents — every Jira ticket, every Linear issue and all 285,605 Slack
messages arrived at the adjudicator undated, so a later correction and the
earlier report it corrected were indistinguishable. Linear, Jira, HubSpot and
Confluence write their dates into the body instead, as activity logs and
revision histories, so `derive_date` takes the latest one found there. Date
coverage went from **26% to 51%** of the corpus, and from 0% to ~97% on exactly
the ticket and CRM documents where `status` and `owner` claims live.

**The important part is that it is allowed to refuse.** When the margin between
two values is too thin, it says so and hands both to the model:

```
DISAGREEMENT — status of PD-INC-9821, 4 competing claims:
  NO ACCEPTED VALUE — trust is too close to choose (0.67 against 0.67)
  Do not choose between these. Report that the documents disagree
  and give both values with their sources.
```

### The contradiction graph

Arbitration used to happen per question and then be thrown away. It is now
written into HydraDB as edges, which changes what can be asked. Three
questions become one traversal each, and none of them requires anyone to have
asked about a document first:

**What does this organisation contradict itself about?** Rank the
`Disagreement` nodes. This is a label scan — the exact shape the engine
rejects on `Entity` and `Document` — and it works here only because the label
is small enough to stay scannable. Keeping it that way is the loader's job.

**What was this value before, and what corrected it?** Walk `SUPERSEDES`
outwards from the current claim. Every hop names the document that changed it.
The edge is written *only* where recency is what actually settled the conflict;
an unresolved disagreement is not the history of a fact, and drawing it as one
would be a claim about time that the evidence does not support.

**Who has been reading the version that turned out to be wrong?** Anchor on a
claim, hop to the document asserting it, hop back out to everyone the ontology
connects to that document — `SENT`, `SPOKE_IN` and `MENTIONED_IN` kept
separate, because sending a document that states a superseded limit is a
different position from being mentioned in it.

Claims are extracted offline over the work items the most tools quote. That
ranking is the corpus narrowing itself rather than us picking: a key like
`ENG-4821` appears across GitHub, Linear, Drive, Confluence and Slack, and that
cross-quotation is the precondition for two documents to disagree at all. A
contradiction inside one Jira ticket is a typo; a contradiction between the
Confluence page and the Slack thread about one work item is an organisation
that has lost track of its own decision.

Two constraints are worth stating plainly because they shaped the design:

- **A disagreement must span two documents.** One meeting transcript listing
  three thresholds is one text read three times. Those claims are still
  written and still queryable — what they do not get is a node on a map
  asserting the company contradicts itself.
- **Nothing in this graph can be deleted.** `DETACH DELETE` is refused by
  admission control even for a single anchored node with two edges, because
  deleting a vertex scans its edges. So a reload cannot replace the previous
  one, only sit beside it. Every load stamps its nodes with a generation and
  the reader asks for one stamp — otherwise the map is an accumulation of
  every map ever built.

---

## Results

Measured with `scripts/grade.py`, which grades every answer against the
benchmark's own `answer_facts` rubric with an LLM judge, one fact at a time.
The answer key lives **outside this repository** and only the two scoring
scripts may read it; nothing under `src/glasshouse` can see it.

Measured 2026-08-20, 20 questions per category. The baseline is the same
harness run on 2026-08-18, before any of the work below.

| Category | Questions | Fact recall |
|---|---:|---:|
| `info_not_found` — knowing the answer is absent | 20 | **100%** |
| `intra_document_reasoning` | 40 | **81%** |
| `constrained` | 30 | **67%** |
| `conflicting_info` — the arbitration case | 20 | **59%** |
| `basic` | 175 | 40% |
| `project_related` | 40 | 38% |
| `metadata` — authorship, ownership, location | 100 | 32% |
| `completeness` | 20 | 32% |
| `semantic` — deliberate paraphrase | 125 | 29% |
| **weighted overall** | **570 / 600** | **≈44%**  (from 35.5%) |

`miscellaneous` and `high_level` (30 questions) are not yet measured.

What moved, and why:

- **`metadata` went from 0% to 32%.** It sat at exactly zero across two
  independent runs. The cause was not the model: the facet fields were
  normalized and then discarded, so the answer to "which space is this page
  in" was never in the index and never shown to the model. Indexing the facets,
  scoping by container and reranking a deep page took the expected document
  reaching the model from 8/30 to 16/30.
- **`conflicting_info` went from ~52% to 59%**, after claims arbitration and
  after recovering the dates that arbitration ranks on.
- **`info_not_found` held at 100%** through every prompt change — including the
  ones that deliberately made the system more willing to answer elsewhere. That
  it did not move is the result.

One change mattered more than any retrieval work. `ask()` used to refuse
outright when no person resolved in the retrieved documents — so "who authored
the SLO throttler PR", whose answer *is* a person and which names none, was
returned as an empty string with the correct document sitting in the context.
`stream()` never had that gate, so the interface answered questions the graded
path silently declined.

<!-- RESULTS -->

### Things we measured that did not work

Kept here because a negative result that cost a day is worth as much as a
feature, and because every one of these is a plausible-sounding idea:

- **Person-seeded graph retrieval.** Opens for 21 of 570 questions, and where it
  added documents it never added a correct one. The obvious graph-RAG design is
  the wrong one for this benchmark.
- **LLM query expansion.** `semantic` questions paraphrase deliberately — "too
  many requests errors" for `429`, "Western Europe" for `eu-west`. Asking a model
  to translate the question into document jargon made retrieval **worse**
  (9/30 → 5/30): it invents plausible identifiers like `safest_numeric_mode`
  that appear in no document, and the noise displaces real hits. Filtering the
  expansion against actual corpus frequency did not rescue it.
- **Reranking order.** Whether the container scope is placed first, middle or
  last changes nothing at any context budget. The budget was the constraint, not
  the ordering.

---

## Getting started

```bash
git clone https://github.com/Shrujal00/glasshouse-hydradb.git
cd glasshouse-hydradb

cp .env.example .env      # add your HydraDB and Ollama keys
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

docker compose up -d      # HydraDB open-source engine
.venv/bin/python -m pytest
```

Then build the indexes and the graph. Each step is restartable and prints what
it did; the timings are measured on the full 511,962-document corpus.

```bash
.venv/bin/python scripts/fetch_corpus.py        # pulls the corpus + answer key
.venv/bin/python scripts/intake.py              # → data/normalized/*.jsonl

# local indexes — no network, no API key
.venv/bin/python scripts/build_index.py         # FTS5 + facets   4.3 GB   ~3.5 min
.venv/bin/python scripts/build_facets.py        # facet store    0.4 GB      ~25s

# the ontology: 209,388 surface forms → 166,429 identities
.venv/bin/python scripts/resolve_entities.py
.venv/bin/python scripts/load_graph.py          # → ontology.sqlite3

# HydraDB: entities and documents first, then containers and roles
.venv/bin/python scripts/load_document_graph.py
.venv/bin/python scripts/load_facet_graph.py    # 1.23M edges     ~20 min
.venv/bin/python scripts/load_surface_graph.py  # 209,388 surfaces

# the contradiction graph — extraction is a model call, so this one costs
.venv/bin/python scripts/load_claims_graph.py --keys 260 --dry-run   # what it would read
.venv/bin/python scripts/load_claims_graph.py --keys 260
```

`load_facet_graph.py` expects the documents and entities to already exist — it
adds containers and edges and never creates an endpoint, so running it before
`load_document_graph.py` writes containers that connect to nothing. The same
holds for `load_claims_graph.py`, which wires claims to documents and people.

`load_claims_graph.py` checkpoints every work item to
`data/state/claims_graph.jsonl` as it goes, so an interrupted run resumes
without paying for the same extraction twice. If a whole run comes back with
no claims, check for a `429` before believing the corpus is quiet — a
rate-limited extraction returns valid JSON asserting nothing, which is
indistinguishable from a document that genuinely says nothing.

Then serve it:

```bash
.venv/bin/python -m uvicorn glasshouse.server:app \
    --host 127.0.0.1 --port 8080 --app-dir src
```

**Requirements:** Docker, Python 3.11+, a [HydraDB](https://app.hydradb.com) API
key (Ship tier is free), and an [Ollama Cloud](https://ollama.com) key.

The system degrades rather than breaks. Without the facet store it retrieves
exactly as it did before that store existed; without the container half of the
graph the local facet table serves the same scope, and the trace says which of
the two answered.

The benchmark corpus is not redistributed here; `fetch_corpus.py` pulls it from
upstream. That script also keeps the benchmark's answer key **outside** this
repository, so the pipeline cannot train or tune against it.

---

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
