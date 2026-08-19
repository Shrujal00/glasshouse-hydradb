"""An in-memory stand-in for the ontology half of HydraDB.

Identity resolution reads `(:Surface)-[:DENOTES]->(:Entity)` out of the engine,
so the tests need something that answers those two traversals. This serves them
from the same `(surface, kind, eid, node_id, canonical_name, confidence,
alias_count)` rows the tests already declare, which keeps every assertion about
*behaviour* and none about storage.

The two corpus-wide counts the real loader precomputes onto the Surface node
are computed here the same way, so a test that turns on ambiguity or on
given-name evidence is exercising the real rule rather than a stub of it.
"""

from __future__ import annotations

from collections.abc import Sequence

from glasshouse.graph import SurfaceMatch


class FakeOntologyGraph:
    """Answers `denoted_by` and `surfaces_of` from a list of alias rows."""

    def __init__(self, aliases: Sequence[tuple]) -> None:
        self.aliases = [tuple(row) for row in aliases]

    # --- the two traversals identity resolution uses ------------------------

    def denoted_by(self, text: str, limit: int = 4) -> list[SurfaceMatch]:
        word = (text or "").strip().lower()
        rows = [r for r in self.aliases if str(r[0]).lower() == word]
        if not rows:
            return []
        # `entities` counts people, not rows: one person written as both a name
        # and an address is one entity, and must not read as ambiguous.
        eids = {r[2] for r in rows}
        given = len(
            {
                str(r[0]).lower()
                for r in self.aliases
                if r[1] == "name" and str(r[0]).lower().startswith(f"{word} ")
            }
        )
        kinds = tuple(sorted({str(r[1]) for r in rows if r[1]}))
        out: list[SurfaceMatch] = []
        for eid in sorted(eids):
            row = next(r for r in rows if r[2] == eid)
            out.append(
                SurfaceMatch(
                    text=word,
                    kinds=kinds,
                    entities=len(eids),
                    given_name_forms=given,
                    eid=str(row[2]),
                    name=str(row[4]),
                    node=int(row[3]),
                    confidence=float(row[5]),
                    alias_count=int(row[6]),
                )
            )
        return out[:limit]

    def surfaces_of(self, entity_node: int, limit: int = 40) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        for row in self.aliases:
            if int(row[3]) == int(entity_node):
                pair = (str(row[0]).lower(), str(row[1]))
                if pair not in found:
                    found.append(pair)
        return found[:limit]

    # --- the rest of the engine, absent by default --------------------------

    def documents_for_entities(self, seeds, limit):
        return []

    def entities_for_documents(self, doc_ids, limit, documents=8):
        return []

    def documents_in_containers(self, containers, limit=200):
        return []
