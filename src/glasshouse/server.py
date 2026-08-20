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
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from .ask import Asker
from .graph import node_id
from .config import STATE

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


@app.get("/api/claims/stats")
def claim_stats() -> dict:
    """Size of the contradiction graph, for the header."""
    out = asker().engine.claim_stats()
    out["generation"] = _generation()
    try:
        raw = json.loads((STATE / "claims_graph.json").read_text())
        out["work_items"] = raw.get("work_items", 0)
        out["undecided"] = raw.get("undecided", 0)
    except Exception:
        pass
    return out


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
    try:
        out["graph"] = asker().engine.claim_stats()
    except Exception:
        pass
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
