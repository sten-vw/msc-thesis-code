"""Retrieval metrics via pytrec_eval."""

from __future__ import annotations

from core.types import RelevanceJudgments


def compute_per_query_retrieval(
    run: dict[str, dict[str, float]],
    qrels: RelevanceJudgments,
    metric: str = "ndcg_cut_10",
) -> dict[str, float]:
    """{query_id: metric} for every query present in `run`."""
    import pytrec_eval

    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: dict(docs) for qid, docs in qrels.judgments.items()}, [metric]
    )
    return {qid: res.get(metric, 0.0) for qid, res in evaluator.evaluate(run).items()}
