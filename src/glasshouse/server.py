"""The web surface: a question box, and the reasoning made watchable.

Server-sent events rather than a single JSON response, because the point of
the interface is that you see the graph being built — documents landing,
aliases collapsing onto people, HydraDB walking between them — rather than
being handed a finished answer and asked to trust it.

Runs entirely locally against the corpus index, the ontology and the HydraDB
engine in Docker. No account, no key, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from functools import lru_cache

from .ask import Asker
from .graph import node_id
from .config import ROOT, STATE

WEB = Path(__file__).parent / "web"

app = FastAPI(title="Glasshouse", docs_url=None, redoc_url=None)
_asker: Asker | None = None


def asker() -> Asker:
    """One Asker for the process, so the indexes stay open between questions."""
    global _asker
    if _asker is None:
        _asker = Asker()
    return _asker


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/api/stats")
def stats() -> dict:
    """Corpus and ontology size, for the header."""
    out: dict = {"documents": 0, "entities": 0, "aliases": 0}
    try:
        out["documents"] = asker().recall.count()
    except Exception:
        pass
    try:
        stats_path = STATE / "resolve_stats.json"
        if stats_path.exists():
            raw = json.loads(stats_path.read_text())
            out["entities"] = raw.get("entities", 0)
            out["aliases"] = raw.get("surfaces", 0)
            out["collapsed"] = raw.get("multi_alias_entities", 0)
    except Exception:
        pass
    return out


@app.get("/api/ask")
def ask(q: str, limit: int = 20) -> StreamingResponse:
    def events():
        try:
            for event in asker().stream(q, limit=limit):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # surface failures in the UI, not the console
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)[:300]})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/entity/{eid}")
def entity(eid: str) -> dict:
    """Every alias of one person, with the evidence that attached it.

    Read from the graph rather than from a file: this is the question the
    ontology exists to answer, and answering it by traversal is the difference
    between a stored result and a queryable one.
    """
    current = asker()
    # The node id is derived from the eid rather than looked up: ids here are a
    # deterministic digest of a stable key, so the address of a person is
    # computable and a table mapping one to the other has nothing to add.
    rows = current.engine.query(
        "MATCH (a:Alias)-[r:RESOLVES_TO]->(e:Entity {id: $id}) "
        "RETURN a.surface AS surface, a.kind AS kind, a.occurrences AS occurrences, "
        "r.score AS score, r.signals AS signals ORDER BY occurrences DESC",
        {"id": node_id(f"entity:{eid}")},
        strong=True,
    )
    # An entity nothing resolves to is one the ontology never built, which is
    # the same answer as one that does not exist.
    if not rows:
        raise HTTPException(status_code=404, detail="entity not found")
    return {"eid": eid, "aliases": [r.values for r in rows]}


# --- the contradiction graph ------------------------------------------------
#
# Everything below reads what `scripts/load_claims_graph.py` wrote. None of it
# asks a question, retrieves a document or calls a model: the reasoning already
# happened, offline, and these are traversals over the result. That is the
# whole point of putting arbitration in the graph rather than in a cache --
# "what does this company contradict itself about?" is answerable without
# anyone having asked about any particular document first.


def _generation() -> str:
    """Which load to read.

    Nothing in this graph can be deleted, so every load leaves the previous
    one behind. The loader stamps its nodes and records the stamp; reading one
    stamp is what makes the map current rather than cumulative.
    """
    try:
        raw = json.loads((STATE / "claims_graph.json").read_text())
        return str(raw.get("gen") or "")
    except Exception:
        return ""


@app.get("/api/disagreements")
def disagreements(limit: int = 40, undecided: bool = False, predicate: str = "") -> dict:
    """The disagreement map: one row per thing the company contradicts itself about."""
    found = asker().engine.disagreements(
        limit=limit,
        undecided_only=undecided,
        predicate=predicate,
        gen=_generation(),
    )
    return {
        "generation": _generation(),
        "disagreements": [
            {
                "key": d.key,
                "scope": d.scope,
                "subject": d.subject,
                "predicate": d.predicate,
                "sides": d.sides,
                "claims": d.claims,
                "documents": d.documents,
                "sources": list(d.sources),
                "trust_gap": round(d.trust_gap, 3),
                "decided": d.decided,
                "winner": {
                    "value": d.winner_value,
                    "source": d.winner_source,
                    "trust": round(d.winner_trust, 3),
                    "claim_id": d.winner_claim_id,
                },
                "runner": {
                    "value": d.runner_value,
                    "source": d.runner_source,
                    "trust": round(d.runner_trust, 3),
                    "claim_id": d.runner_claim_id,
                },
                "rationale": d.rationale,
                "first_asserted": d.first_asserted,
                "last_asserted": d.last_asserted,
                "entity": {"eid": d.entity_eid, "name": d.entity_name}
                if d.entity_eid
                else None,
                "weight": round(d.weight, 2),
            }
            for d in found
        ],
    }


@app.get("/api/disagreement/{key}")
def disagreement(key: str) -> dict:
    """One disagreement, every claim on every side, and what settled it."""
    head, claims = asker().engine.disagreement(key, gen=_generation())
    if head is None:
        raise HTTPException(status_code=404, detail="disagreement not found")
    return {
        "key": head.key,
        "scope": head.scope,
        "subject": head.subject,
        "predicate": head.predicate,
        "decided": head.decided,
        "rationale": head.rationale,
        "sides": head.sides,
        "sources": list(head.sources),
        # `decided` is false when arbitration refused to choose. The interface
        # must not draw that as a verdict -- refusing is a result, and dressing
        # it as a winner is the exact failure the trust floor exists to stop.
        "winner_claim_id": head.winner_claim_id if head.decided else None,
        "entity": {"eid": head.entity_eid, "name": head.entity_name}
        if head.entity_eid else None,
        "claims": [
            {
                "claim_id": c.claim_id,
                "value": c.object_value,
                "subject": c.subject,
                "predicate": c.predicate,
                "source": c.source,
                "date": c.asserted_at,
                "trust": round(c.trust, 3),
                "status": c.status,
                "doc_id": c.doc_id,
                "title": c.title,
                "cite": c.cite,
                "side": c.side,
            }
            for c in claims
        ],
    }


@app.get("/api/claim/{claim_id}/history")
def claim_history(claim_id: str) -> dict:
    """What this value used to be, and what corrected it — one SUPERSEDES walk."""
    chain = asker().engine.claim_history(claim_id)
    return {
        "claim_id": claim_id,
        "chain": [
            {
                "claim_id": step.get("claim_id", ""),
                "value": step.get("object_value", ""),
                "subject": step.get("subject", ""),
                "predicate": step.get("predicate", ""),
                "source": step.get("source", ""),
                "date": step.get("asserted_at", ""),
                "trust": step.get("trust", 0.0),
                "status": step.get("status", ""),
                "doc_id": step.get("doc_id", ""),
                "title": step.get("title", ""),
            }
            for step in chain
        ],
    }


@app.get("/api/claim/{claim_id}/blast")
def claim_blast(claim_id: str, limit: int = 60) -> dict:
    """Who has been reading this version — Claim → Document → the people on it."""
    reached = asker().engine.blast_radius(claim_id, limit=limit)
    return {
        "claim_id": claim_id,
        "people": reached,
        "documents": sorted({r["doc_id"] for r in reached}),
    }


@app.get("/api/entity/{eid}/claims")
def entity_claims(eid: str, limit: int = 30) -> dict:
    """Every claim wired to one person, through `ABOUT`.

    This is the join between the two halves of the product. The ontology knows
    who someone is; the contradiction graph knows what is asserted about them.
    Until now nothing walked from one to the other, so they read as two
    separate demos rather than one system.
    """
    found = asker().engine.claims_about(eid, limit=limit)
    return {
        "eid": eid,
        "claims": [
            {
                "claim_id": c.claim_id,
                "scope": c.scope,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.object_value,
                "source": c.source,
                "date": c.asserted_at,
                "trust": round(c.trust, 3),
                "status": c.status,
                "title": c.title,
                "doc_id": c.doc_id,
            }
            for c in found
        ],
    }


@app.get("/api/claims/stats")
def claim_stats() -> dict:
    """Size of the contradiction graph, for the header.

    Same reason as `/api/ontology`: `MATCH (a:Claim)-[:CONTRADICTS]->(b:Claim)
    RETURN count(*)` takes about 17 seconds on this graph, and the header is
    not worth a 34-second request.
    """
    out: dict = {"generation": _generation()}
    try:
        raw = json.loads((STATE / "claims_graph.json").read_text())
        out.update(raw.get("graph") or {})
        out["work_items"] = raw.get("work_items", 0)
        out["undecided"] = raw.get("undecided", 0)
    except Exception:
        try:
            out.update(asker().engine.claim_stats())
        except Exception:
            pass
    return out


# --- how a benchmark question was graded ------------------------------------
#
# This does NOT read the answer key, and it must never be made to. The rule the
# whole measurement rests on is that nothing under `src/glasshouse` can see
# `$GOLD_ANSWERS_PATH` -- a pipeline that can read what it is marked against is
# a pipeline whose score means nothing.
#
# Two files are read here and neither is the key. `questions_blind.jsonl` is
# the published question list: ids and question text, no answers. And
# `answer_grade.json` is what `scripts/grade.py` wrote *after* the fact --
# `{question_id, type, facts, supported, seconds}`, counts and nothing else.
# Neither can tell the system what the right answer is; together they can tell
# a reader that the answer they are looking at was independently marked, and
# what it scored.


@lru_cache(maxsize=1)
def _benchmark() -> dict[str, dict]:
    """Question text → the grade `scripts/grade.py` recorded for it."""
    blind = ROOT / "data" / "bench" / "questions_blind.jsonl"
    ids: dict[str, str] = {}
    try:
        for line in blind.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids[_normal(row.get("question", ""))] = row.get("question_id", "")
    except Exception:
        return {}
    graded: dict[str, dict] = {}
    try:
        report = json.loads((STATE / "answer_grade.json").read_text())
        by_id = {row["question_id"]: row for row in report.get("detail", [])}
    except Exception:
        return {}
    for text, qid in ids.items():
        row = by_id.get(qid)
        if row:
            graded[text] = {
                "question_id": qid,
                "type": row.get("type", ""),
                "facts": row.get("facts", 0),
                "supported": row.get("supported", 0),
                "seconds": row.get("seconds"),
            }
    return graded


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()


@app.get("/api/graded")
def graded(q: str) -> dict:
    """What an independent judge scored this answer, if it is a benchmark question.

    Marked one required fact at a time by `scripts/grade.py`, against a rubric
    this process cannot read.
    """
    row = _benchmark().get(_normal(q))
    if not row:
        return {"graded": False}
    return {"graded": True, **row}


# --- the ontology -----------------------------------------------------------


@app.get("/api/ontology")
def ontology() -> dict:
    """What the ontology is, and what building it cost.

    Track 01 asks for an ontology, and until now the only way to see one was
    to read `resolve_stats.json` off disk. The single most telling number in
    there is `merges_refused_by_constraint`: the resolver refused nearly four
    times more merges than it made, which is the difference between an
    ontology and a pile of string-similarity guesses.
    """
    out: dict = {"schema": [], "resolve": {}, "graph": {}}
    try:
        raw = json.loads((STATE / "resolve_stats.json").read_text())
        out["resolve"] = {
            key: raw.get(key)
            for key in (
                "raw_surfaces", "surfaces", "candidate_pairs",
                "pairs_above_threshold", "merges_applied",
                "merges_refused_by_constraint", "entities",
                "multi_alias_entities", "shared_mailboxes_excluded",
                "threshold", "min_occurrences", "resolve_seconds", "docs",
            )
            if raw.get(key) is not None
        }
    except Exception:
        pass
    # Read from what the loader recorded, not by counting. Counting
    # relationships costs ~17s per edge type on the loaded graph -- two of
    # them together blow past any request timeout and the page just hangs.
    # The graph cannot be written to except by the loader, so the number the
    # loader wrote is the number, and the engine is only asked when that
    # record is missing.
    try:
        out["graph"] = json.loads((STATE / "claims_graph.json").read_text())["graph"]
    except Exception:
        try:
            out["graph"] = asker().engine.claim_stats()
        except Exception:
            out["graph"] = {}
    # Written out rather than read back from the engine: counting `Entity` or
    # `Document` is the unanchored scan that times out or returns 429, so the
    # sizes come from the loaders that wrote them.
    resolved = out["resolve"]
    out["schema"] = [
        {"pattern": "(:Surface)-[:DENOTES]->(:Entity)",
         "count": resolved.get("surfaces"),
         "note": "a word from a question, resolved in one hop"},
        {"pattern": "(:Alias)-[:RESOLVES_TO {score, signals}]->(:Entity)",
         "count": resolved.get("merges_applied"),
         "note": "every accepted merge, with the evidence for it"},
        {"pattern": "(:Entity)-[:MENTIONED_IN|SPOKE_IN|SENT]->(:Document)",
         "count": resolved.get("docs"),
         "note": "who is named in, spoke in, or sent each document"},
        {"pattern": "(:Document)-[:IN_CONTAINER]->(:Container)",
         "count": 978512,
         "note": "the channel, folder or space a document lives in"},
        {"pattern": "(:Claim)-[:CONTRADICTS]->(:Claim)",
         "count": out["graph"].get("contradicts"),
         "note": "two documents asserting different values for one thing"},
        {"pattern": "(:Claim)-[:SUPERSEDES]->(:Claim)",
         "count": out["graph"].get("supersedes"),
         "note": "the value that replaced an earlier one, and when"},
    ]
    return out


@app.get("/api/who/{text}")
def who(text: str) -> dict:
    """Who one written form denotes — the identity traversal, exposed.

    This is `Surface -[:DENOTES]-> Entity` and nothing else: one anchored hop
    on a node id derived from the word itself. A form reaching more than one
    person has not named anybody, and that is reported rather than resolved,
    because picking one would be the guess the whole resolver exists to avoid.
    """
    current = asker()
    matches = current.engine.denoted_by(text, limit=8)
    people = []
    for match in matches:
        aliases = []
        try:
            rows = current.engine.query(
                "MATCH (a:Alias)-[r:RESOLVES_TO]->(e:Entity {id: $id}) "
                "RETURN a.surface AS surface, a.kind AS kind, "
                "a.occurrences AS occurrences, r.score AS score, "
                "r.signals AS signals ORDER BY occurrences DESC",
                {"id": node_id(f"entity:{match.eid}")},
                strong=True,
            )
            aliases = [
                {
                    "surface": r.values.get("surface"),
                    "kind": r.values.get("kind"),
                    "occurrences": r.values.get("occurrences"),
                    "score": r.values.get("score"),
                    "signals": r.values.get("signals"),
                }
                for r in rows
            ]
        except Exception:
            pass
        people.append({
            "eid": match.eid,
            "name": match.name,
            "confidence": round(match.confidence, 3),
            "alias_count": match.alias_count,
            "written_as": [a for a in aliases if a.get("surface")],
        })
    return {
        "text": text,
        "denotes": len(matches),
        # The ambiguity guard, said plainly. `sam` reaches eight people in this
        # corpus and none of them has been named.
        "ambiguous": len(matches) > 1 or (matches[0].entities > 1 if matches else False),
        "people": people,
    }
