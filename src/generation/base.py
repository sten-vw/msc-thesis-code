"""Shared helpers for synthetic query generation strategies."""

from __future__ import annotations

import uuid

from core.types import Document, Query


def make_query(text: str, source_doc: Document, strategy: str) -> Query:
    """A synthetic query carrying its source document and generating strategy."""
    return Query(
        query_id=f"syn_{uuid.uuid4().hex[:12]}",
        text=text.strip(),
        source_doc_id=source_doc.doc_id,
        generation_strategy=strategy,
    )
