"""Build one synthetic query pool from a named strategy."""

from __future__ import annotations

import math
import random

import pandas as pd

from core.llm import BedrockLLM
from core.types import Document, Query
from generation import STRATEGIES


def sample_documents(
    corpus: dict[str, Document], n_queries: int, queries_per_doc: int, seed: int
) -> list[Document]:
    documents = list(corpus.values())
    limit = math.ceil(n_queries / max(queries_per_doc, 1))
    if limit >= len(documents):
        return documents
    return random.Random(seed).sample(documents, limit)


def build_generator(
    strategy: str, llm: BedrockLLM, corpus_name: str, seed: int, **overrides
):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {sorted(STRATEGIES)}")
    cls = STRATEGIES[strategy]
    kwargs: dict = {"llm": llm}
    if strategy in ("promptagator", "duqgen", "dragon"):
        kwargs["seed"] = seed
    if strategy == "promptagator":
        kwargs["dataset_name"] = corpus_name
    if strategy == "udapdr":
        kwargs = {"seed": seed}
    kwargs.update(overrides)
    return cls(**kwargs)


def generate_pool(
    strategy: str,
    corpus: dict[str, Document],
    corpus_name: str,
    llm: BedrockLLM,
    *,
    seed: int,
    n_queries: int,
    queries_per_doc: int = 1,
    train_queries: list[Query] | None = None,
    **overrides,
) -> list[Query]:
    """``n_queries`` synthetic queries over documents drawn at ``seed``; DUQGen instead
    receives the whole corpus.
    """
    generator = build_generator(strategy, llm, corpus_name, seed, **overrides)
    if strategy == "duqgen":
        return generator.generate(
            list(corpus.values()),
            few_shot_queries=train_queries,
            num_queries=n_queries,
            num_queries_per_doc=queries_per_doc,
        )
    documents = sample_documents(corpus, n_queries, queries_per_doc, seed)
    if strategy == "promptagator":
        return generator.generate(
            documents,
            few_shot_queries=train_queries or [],
            few_shot_docs=corpus,
            num_queries_per_doc=queries_per_doc,
        )
    return generator.generate(documents)


def pool_to_frame(queries: list[Query], limit: int | None = None) -> pd.DataFrame:
    rows = [
        {"query_id": q.query_id, "text": q.text, "source_doc_id": q.source_doc_id}
        for q in queries
        if q.source_doc_id
    ]
    if limit is not None:
        rows = rows[:limit]
    return pd.DataFrame(rows, columns=["query_id", "text", "source_doc_id"])


def frame_to_queries(df: pd.DataFrame) -> list[Query]:
    return [
        Query(query_id=r.query_id, text=r.text, source_doc_id=r.source_doc_id)
        for r in df.itertuples()
    ]
