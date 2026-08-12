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

> **Build in progress.** This project is being developed for Hack Hydra,
> 12–20 August 2026. Full documentation — architecture, ontology design,
> evaluation results and demo — lands with the final submission on **20 August**.

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

## Getting started

```bash
git clone https://github.com/Shrujal00/glasshouse-hydradb.git
cd glasshouse-hydradb

cp .env.example .env      # add your HydraDB and Ollama keys
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

docker compose up -d      # HydraDB open-source engine
python scripts/fetch_corpus.py
python scripts/intake.py
.venv/bin/python -m pytest
```

**Requirements:** Docker, Python 3.11+, a [HydraDB](https://app.hydradb.com) API
key (Ship tier is free), and an [Ollama Cloud](https://ollama.com) key.

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
