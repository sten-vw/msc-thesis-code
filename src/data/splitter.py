"""Deterministic train/test split of real queries."""

from __future__ import annotations

import random

from core.types import Query, RelevanceJudgments


def split_queries(
    queries: dict[str, Query],
    qrels: RelevanceJudgments,
    train_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, Query], dict[str, Query], RelevanceJudgments, RelevanceJudgments]:
    rng = random.Random(seed)
    qids = sorted(queries.keys())
    rng.shuffle(qids)

    split_idx = int(len(qids) * train_ratio)
    train_qids = set(qids[:split_idx])
    test_qids = set(qids[split_idx:])

    train_queries = {qid: queries[qid] for qid in train_qids}
    test_queries = {qid: queries[qid] for qid in test_qids}
    train_qrels = RelevanceJudgments(
        judgments={q: j for q, j in qrels.judgments.items() if q in train_qids}
    )
    test_qrels = RelevanceJudgments(
        judgments={q: j for q, j in qrels.judgments.items() if q in test_qids}
    )
    return train_queries, test_queries, train_qrels, test_qrels
