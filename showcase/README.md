<div align="center">

# 🏠 Glasshouse

**An enterprise ontology you can see through.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](../pyproject.toml)
[![Built on HydraDB](https://img.shields.io/badge/built%20on-HydraDB-ff5c39.svg)](https://hydradb.com)
[![Hack Hydra 2026](https://img.shields.io/badge/Hack%20Hydra-2026-6e56cf.svg)](https://hackhydra.hydradb.com)

</div>

---

## What is Glasshouse?

Glasshouse resolves the mess that happens when nine enterprise tools disagree about who owns what.

**511,962 documents. Nine sources. One graph.**

Ask a question in English and watch the reasoning happen — identities resolving, sources disagreeing, and the system declining to answer when evidence doesn't support a verdict.

---

## What it does

| # | Capability | What happens |
|:-:|---|---|
| 1 | **Identity Resolution** | 401,163 name forms → 166,429 unique people. 163,262 merges *refused* by hard rules. |
| 2 | **Claim Extraction** | Typed claims with subject, value, source, and date — extracted only from shown text. |
| 3 | **Deterministic Arbitration** | Source authority, recency, corroboration. Same input → same output, every time. Can refuse when unsure. |

---

## How HydraDB powers it

### 🔶 Open-source engine (self-hosted)
The ontology and contradiction graph — 209,388 surfaces, 978,512 edges.

### 🔶 Managed cloud
Hybrid dense + sparse retrieval for document recall.

**Turn the engine off and all queries return zero.** HydraDB is where the reasoning lives, not just storage.

---

## Quick start

```bash
git clone https://github.com/Shrujal00/glasshouse-hydradb.git
cd glasshouse-hydradb

python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env        # fill in your keys
docker compose up -d        # start HydraDB engine
.venv/bin/python -m pytest  # 183 tests, no network needed
```

Then run:

```bash
.venv/bin/python -m uvicorn glasshouse.server:app \
    --host 127.0.0.1 --port 8080 --app-dir src
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) → lands on the disagreement map.

---

## Results

| Category | Fact Recall |
|---|---:|
| Info not found (knowing absence) | **100%** |
| Intra-document reasoning | **81%** |
| Constrained queries | **67%** |
| Conflicting info (arbitration) | **60%** |
| **Weighted overall** | **≈44%** (from 35.5%) |

---

## Tech stack

- **Python 3.11+** — no framework
- **HydraDB** — open-source engine + cloud
- **SQLite FTS5** — BM25 search over 511K docs
- **FastAPI + SSE** — streaming reasoning
- **Vanilla HTML/CSS/JS** — no build step
- **183 tests** — no network required

---

## Built by

**[Shrujal Ganatra](https://github.com/Shrujal00)** for [Hack Hydra 2026](https://hackhydra.hydradb.com)

[LinkedIn](https://www.linkedin.com/in/shrujal-ganatra/) · Apache-2.0 License
