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
