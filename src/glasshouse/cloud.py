"""HydraDB managed cloud — the recall layer.

Glasshouse uses HydraDB twice, for two different jobs. This module is the
cloud half: given a question, hand back the handful of documents worth
reading out of half a million. The graph half lives in `graph.py` and runs
on the self-hosted open-source engine.

The metadata schema has to exist before `document_metadata` is accepted on
ingest, so `ensure_database()` is not optional setup — skip it and every
upload silently loses its metadata, or fails with a 400.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import hydra_db
from hydra_db.errors.conflict_error import ConflictError
from hydra_db.types.tenants_custom_property_definition import (
    TenantsCustomPropertyDefinition as PropertyDef,
)

from .config import HYDRA_CLOUD_DATABASE, require

# Fields we filter and rank on. `source` and `date` earn their keep in the
# query planner: source drives per-tool trust, date drives recency arbitration
# between two documents that contradict each other.
METADATA_SCHEMA: tuple[PropertyDef, ...] = (
    PropertyDef(name="doc_id", data_type="VARCHAR", max_length=64, enable_match=True),
    PropertyDef(name="source", data_type="VARCHAR", max_length=32, enable_match=True),
    PropertyDef(name="title", data_type="VARCHAR", max_length=512, enable_analyzer=True),
    PropertyDef(name="date", data_type="VARCHAR", max_length=10, enable_match=True),
    PropertyDef(name="ticket_key", data_type="VARCHAR", max_length=32, enable_match=True),
    PropertyDef(name="thread_id", data_type="VARCHAR", max_length=128, enable_match=True),
)


@dataclass(slots=True)
class Hit:
    doc_id: str
    score: float
    source: str
    title: str
    text: str

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> "Hit":
        """One chunk of the v2 retrieval result.

        The field names moved: results arrive under `chunks`, the text is
        `chunk_content`, the score is `relevancy_score`, and our own ingest
        metadata is under `additional_metadata`. The old names are kept as
        fallbacks so a payload from either shape still parses rather than
        silently producing a hit with no text.
        """
        meta = (
            raw.get("additional_metadata")
            or raw.get("document_metadata")
            or raw.get("metadata")
            or {}
        )
        return cls(
            doc_id=str(meta.get("doc_id") or raw.get("id") or ""),
            score=float(
                raw.get("relevancy_score") or raw.get("score") or raw.get("relevance") or 0.0
            ),
            source=str(meta.get("source") or raw.get("source_type") or ""),
            title=str(meta.get("title") or raw.get("source_title") or ""),
            text=str(
                raw.get("chunk_content") or raw.get("content") or raw.get("text") or ""
            ),
        )


class CloudRecall:
    """Thin wrapper over the v2 SDK, holding the conventions in one place."""

    def __init__(self, database: str | None = None, token: str | None = None) -> None:
        self.database = database or HYDRA_CLOUD_DATABASE
        self.client = hydra_db.HydraDB(token=token or require("HYDRA_DB_API_KEY"), timeout=120.0)

    # --- setup --------------------------------------------------------------

    def ensure_database(self) -> str:
        """Create the database with its metadata schema, or patch an existing one.

        Returns a short word describing what happened, so callers can log it.

        The patch is not a no-op when a field is already declared - the API
        answers 409 rather than ignoring it - so a re-run of an unchanged
        schema conflicts with itself. That case is indistinguishable from a
        real type mismatch in the response, and both are caught here: a true
        mismatch surfaces on the next ingest as a rejected batch, which is
        where it is actually diagnosable.
        """
        existing = set(self.client.databases.list().data.databases or [])
        if self.database not in existing:
            self.client.databases.create(
                database=self.database,
                database_metadata_schema=list(METADATA_SCHEMA),
            )
            return "created"
        try:
            self.client.databases.update_metadata_schema(
                self.database, add_fields=list(METADATA_SCHEMA)
            )
        except ConflictError:
            return "schema already declared"
        return "updated"

    def stats(self) -> dict[str, Any]:
        return self.client.databases.stats(database=self.database).data.dict()

    # --- write --------------------------------------------------------------

    def ingest(self, docs: Sequence[dict[str, Any]], collection: str | None = None) -> Any:
        """Upload a batch of normalized documents.

        One multipart file per document, plus a metadata array in the same
        order. Our own fields have to sit under `additional_metadata` - the
        top level only accepts the API's own reserved keys, and sending a bare
        field there is a 400, not a silent drop.

        Returns a 202: the documents are queued, not yet searchable. Use
        `indexed_count()` to watch them land.
        """
        return self.client.context.ingest(
            database=self.database,
            collection=collection,
            type="knowledge",
            upsert="true",
            documents=[
                (f"{d['doc_id']}.txt", doc_text(d).encode(), "text/plain") for d in docs
            ],
            document_metadata=json.dumps(
                [{"id": d["doc_id"], "additional_metadata": _clean_meta(d)} for d in docs]
            ),
        )

    def indexed_count(self) -> int:
        return int(self.stats().get("knowledge_collection", {}).get("row_count") or 0)

    # --- read ---------------------------------------------------------------

    def search(
        self,
        question: str,
        limit: int = 20,
        *,
        mode: str = "hybrid",
        graph_context: bool = False,
        **filters: Any,
    ) -> list[Hit]:
        """Documents worth reading, ranked by the managed service.

        `max_results` rather than `limit`: the SDK renamed the parameter and
        the old call raised `TypeError` on every search, which is why nothing
        in the query path used this module.

        `query_by="hybrid"` is the reason this half exists at all. The local
        index is BM25 and nothing else, so a question that paraphrases -- "too
        many requests errors" for `429` -- has no lexical overlap to match on.
        The managed service brings its own embeddings, which is dense retrieval
        this stack could not otherwise have: every Ollama embedding model
        returns 401 on this account and the GPU is 6 GB.
        """
        response = self.client.query(
            database=self.database,
            query=question,
            max_results=limit,
            query_by=mode,
            graph_context=graph_context,
            **filters,
        )
        data = response.data if hasattr(response, "data") else response
        raw = (
            getattr(data, "chunks", None)
            or getattr(data, "results", None)
            or getattr(data, "documents", None)
            or []
        )
        out = [
            Hit.from_payload(
                r if isinstance(r, dict)
                else (r.model_dump() if hasattr(r, "model_dump") else r.dict())
            )
            for r in raw
        ]
        # One hit per document. The service returns chunks, and three chunks of
        # one page is one document worth reading, not three.
        seen: set[str] = set()
        unique: list[Hit] = []
        for hit in out:
            if hit.doc_id and hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            unique.append(hit)
        return unique


def doc_text(doc: dict[str, Any]) -> str:
    """Title and body as one retrievable blob.

    The title carries real signal in this corpus - Jira summaries and email
    subject lines often state the fact the body only alludes to - so it is
    prepended rather than kept as metadata alone.
    """
    title = (doc.get("title") or "").strip()
    body = (doc.get("body") or "").strip()
    return f"{title}\n\n{body}".strip() if title else body


def _clean_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Keep only declared schema fields, dropping empties.

    Undeclared keys are rejected by the API rather than ignored, so this is a
    guard against a parser gaining a field and quietly breaking ingest.
    """
    allowed = {p.name for p in METADATA_SCHEMA}
    return {k: v for k, v in doc.items() if k in allowed and v not in (None, "", [])}


def batched(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
