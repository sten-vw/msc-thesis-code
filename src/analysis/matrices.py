"""Per-query score matrices and per-pipeline metric aggregation.

Bridges retrieval runs to the (query x pipeline) matrices and {pipeline:
score} vectors the correlation and bootstrap estimators consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.types import RelevanceJudgments
from metrics.retrieval import compute_per_query_retrieval
from settings import PRIMARY_METRIC


def load_runs_per_pipeline(parquet: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load a runs parquet into {pipeline: {query_id: {doc_id: score}}}."""
    df = pd.read_parquet(parquet)
    runs: dict = {}
    for ret, g in df.groupby("retriever"):
        runs[ret] = {
            qid: dict(zip(gg["doc_id"], gg["score"].astype(float)))
            for qid, gg in g.groupby("query_id")
        }
    return runs


def per_query_ndcg(
    runs_df: pd.DataFrame, roster: list[str], qrels: RelevanceJudgments,
    metric: str = PRIMARY_METRIC,
) -> dict[str, dict[str, float]]:
    """Per-pipeline {query_id: metric} from a flat runs DataFrame and qrels."""
    out: dict = {}
    for p in roster:
        g = runs_df[runs_df.retriever == p]
        run = {q: dict(zip(gg.doc_id, gg.score.astype(float)))
               for q, gg in g.groupby("query_id")}
        out[p] = compute_per_query_retrieval(run, qrels, metric)
    return out


def pipe_means(
    runs: dict, qrels: RelevanceJudgments, roster: list[str], metric: str,
    qsubset: set[str] | None = None,
) -> dict[str, float]:
    """Fixed query denominator (`qsubset` or every judged query); a pipeline missing a query
    counts as 0.0 rather than being dropped, so the mean isn't inflated by empty result
    sets.
    """
    qids = list(qsubset) if qsubset is not None else \
        [q for q, docs in qrels.judgments.items() if docs]
    out: dict = {}
    for p in roster:
        per = compute_per_query_retrieval(runs[p], qrels, metric)
        out[p] = float(np.mean([per.get(q, 0.0) for q in qids])) if qids else 0.0
    return out


def query_discriminativity(M: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query `std` (spread across pipelines) and `item_total` (Pearson r vs the roster mean
    profile), from a (queries x pipelines) matrix.
    """
    std = M.std(axis=1)
    means = M.mean(axis=0)
    Mc = M - M.mean(axis=1, keepdims=True)
    dc = means - means.mean()
    denom = np.linalg.norm(Mc, axis=1) * np.linalg.norm(dc)
    it = np.where(denom > 0, (Mc @ dc) / np.where(denom == 0, 1, denom), 0.0)
    return {"std": std, "item_total": it}
