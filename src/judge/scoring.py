"""Batch RAGAS scoring: threaded per-pair judging with a resume-safe JSONL
cache, reference-answer synthesis, per-pipeline aggregation, and correlation
against a real-query retrieval-quality target.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import kendalltau, pearsonr

from core.llm import BedrockLLM
from core.types import Document
from judge.ragas import RagasJudge

logger = logging.getLogger(__name__)

ALL_DIMS = [
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "answer_correctness",
    "answer_similarity",
]
REF_BASED_DIMS = ["answer_correctness", "answer_similarity"]
REF_FREE_DIMS = ["faithfulness", "answer_relevance", "context_relevance"]

# ARES-style reference synthesis (Saad-Falcon et al. 2024)
SYNTHESIS_SYSTEM = (
    "You write reference answers for a question-answering dataset. Each "
    "answer must be supported by the provided source passage. Do not add "
    "facts that are not in the passage. Keep answers to 1-3 sentences."
)
SYNTHESIS_PROMPT = """\
Source passage:
{passage}

Question: {question}

Write a faithful, concise answer to the question using only information \
from the source passage above. If the passage does not contain enough \
information to answer the question, respond with exactly:
INSUFFICIENT_INFORMATION

Answer:"""
_SYNTH_MAX_PASSAGE_CHARS = 16_000


def synthesize_reference_answers(
    queries: dict[str, str],
    source_docs: dict[str, str],
    corpus: dict[str, Document],
    llm: BedrockLLM,
    cache_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """Resume-safe JSONL cache of {query_id, reference_answer}; a query with no source doc, or
    an INSUFFICIENT_INFORMATION response, gets an empty-string reference (reference-based
    dims then default to 0.0). Use a different model family from the judge, or self-
    consistency inflates the scores.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    refs: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                refs[obj["query_id"]] = obj["reference_answer"]

    todo: list[str] = []
    prompts: list[list] = []
    missing: list[str] = []
    for qid, qtext in queries.items():
        if qid in refs:
            continue
        src = corpus.get(source_docs.get(qid, "") or "")
        if src is None:
            missing.append(qid)
            continue
        passage = (src.title + "\n" + src.text) if src.title else src.text
        if len(passage) > _SYNTH_MAX_PASSAGE_CHARS:
            passage = passage[:_SYNTH_MAX_PASSAGE_CHARS]
        prompts.append([
            BedrockLLM.user_message(
                SYNTHESIS_PROMPT.format(passage=passage, question=qtext)
            )
        ])
        todo.append(qid)

    if not todo and not missing:
        return refs

    responses = llm.invoke_batch(
        prompts,
        system=SYNTHESIS_SYSTEM,
        temperature=0.0,
        max_tokens=300,
        on_progress=on_progress,
    ) if todo else []

    with open(cache_path, "a") as fh:
        for qid, resp in zip(todo, responses):
            text = resp.text.strip()
            if text.upper().startswith("INSUFFICIENT_INFORMATION"):
                text = ""
            fh.write(json.dumps({"query_id": qid, "reference_answer": text}) + "\n")
            refs[qid] = text
        for qid in missing:
            fh.write(json.dumps({"query_id": qid, "reference_answer": ""}) + "\n")
            refs[qid] = ""

    return refs


def _build_context(
    doc_ids_scores: list[tuple[str, float]],
    corpus: dict[str, Document],
    max_chars: int,
) -> str:
    """Splits `max_chars` round-robin across the retrieved docs (floor 200 chars each) so the
    first doc can't crowd out the rest.
    """
    docs: list[Document] = []
    for doc_id, _score in doc_ids_scores:
        doc = corpus.get(doc_id)
        if doc:
            docs.append(doc)

    if not docs:
        return ""

    per_doc_budget = max(200, max_chars // len(docs))
    parts: list[str] = []
    any_truncated = False
    for doc in docs:
        text = f"{doc.title}\n{doc.text}" if doc.title else doc.text
        if len(text) > per_doc_budget:
            text = text[:per_doc_budget]
            any_truncated = True
        parts.append(text)

    if any_truncated:
        logger.debug(
            "Retrieved context truncated: %d docs, per-doc budget %d chars (total budget %d).",
            len(docs), per_doc_budget, max_chars,
        )

    return "\n\n".join(parts)


def score_batch_cached(
    queries: dict[str, str],
    retrieval: dict[str, dict[str, list[tuple[str, float]]]],
    answers: dict[str, dict[str, str]],
    source_docs: dict[str, str],
    corpus: dict[str, Document],
    pipelines: list[str],
    llm: BedrockLLM,
    cache_path: Path,
    top_k: int = 5,
    max_context_chars: int = 48000,
    references: dict[str, str] | None = None,
    dimensions: list[str] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Resume-safe JSONL cache keyed by (pipeline, query_id); uses `references` as the
    reference answer when given, else the source doc's text, else 0.0 for both reference-
    based dims.
    """
    dims = dimensions or ALL_DIMS
    unsupported = [d for d in dims if d not in ALL_DIMS]
    if unsupported:
        logger.warning("Unsupported RAGAS dimensions %s; supported: %s", unsupported, ALL_DIMS)

    cache: dict[str, dict[str, dict[str, float]]] = {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                cache.setdefault(obj["pipeline"], {})[obj["query_id"]] = obj["scores"]

    needed = [
        (p, q)
        for p in pipelines
        for q in queries
        if q in answers.get(p, {}) and q not in cache.get(p, {})
    ]

    judge = RagasJudge(llm=llm)

    if not needed:
        return cache

    if "answer_similarity" in ALL_DIMS:
        judge._get_embedder()

    missing_ref_warned = [False]

    def _score_one(pair: tuple[str, str]) -> dict:
        pipeline, qid = pair
        doc_ids_scores = sorted(
            retrieval.get(pipeline, {}).get(qid, []), key=lambda x: -x[1]
        )
        if references is not None:
            ref_answer = references.get(qid) or None
        else:
            source_doc = corpus.get(source_docs.get(qid, "") or "")
            ref_answer = source_doc.text if source_doc else None

        question = queries[qid]
        answer = answers[pipeline][qid]
        context = _build_context(doc_ids_scores[:top_k], corpus, max_context_chars)

        scores: dict[str, float] = {}
        if "faithfulness" in dims:
            scores["faithfulness"] = judge.faithfulness(question, answer, context)
        if "answer_relevance" in dims:
            scores["answer_relevance"] = judge.answer_relevance(question, answer)
        if "context_relevance" in dims:
            scores["context_relevance"] = judge.context_relevance(question, context)

        need_ref = any(d in dims for d in REF_BASED_DIMS)
        if need_ref and not ref_answer and not missing_ref_warned[0]:
            logger.warning(
                "ragas scoring: no reference answer for query %s (and possibly "
                "others); reference-based dims default to 0.0",
                qid,
            )
            missing_ref_warned[0] = True

        if "answer_correctness" in dims:
            scores["answer_correctness"] = (
                judge.answer_correctness(question, answer, ref_answer) if ref_answer else 0.0
            )
        if "answer_similarity" in dims:
            scores["answer_similarity"] = (
                judge.answer_similarity(answer, ref_answer) if ref_answer else 0.0
            )

        return {"pipeline": pipeline, "query_id": qid, "scores": scores}

    workers = max(getattr(llm, "_max_workers", 8), 1)
    lock = threading.Lock()
    done = 0
    with open(cache_path, "a") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_one, pair): pair for pair in needed}
        for future in as_completed(futures):
            obj = future.result()
            with lock:
                fh.write(json.dumps(obj) + "\n")
                fh.flush()
                cache.setdefault(obj["pipeline"], {})[obj["query_id"]] = obj["scores"]
                done += 1
                if on_progress:
                    on_progress(done, len(needed))

    return cache


def load_score_cache(cache_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Replay a `score_batch_cached` JSONL cache without re-scoring."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    if not Path(cache_path).exists():
        return out
    for line in Path(cache_path).read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            out.setdefault(obj["pipeline"], {}).setdefault(obj["query_id"], {}).update(
                obj["scores"]
            )
    return out


def load_reference_cache(cache_path: Path) -> dict[str, str]:
    """Replay a `synthesize_reference_answers` JSONL cache."""
    out: dict[str, str] = {}
    if not Path(cache_path).exists():
        return out
    for line in Path(cache_path).read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            out[obj["query_id"]] = obj["reference_answer"]
    return out


def aggregate_per_pipeline(
    scores: dict[str, dict[str, dict[str, float]]],
    pipelines: list[str],
    query_ids: set[str],
    dims: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Mean RAGAS score per pipeline per dimension, from score_batch_cached output."""
    if dims is None:
        dims = ALL_DIMS

    agg: dict[str, dict[str, float]] = {}
    for p in pipelines:
        p_scores = scores.get(p, {})
        agg[p] = {}
        for dim in dims:
            vals = [
                float(p_scores[q][dim])
                for q in query_ids
                if q in p_scores and dim in p_scores[q]
            ]
            agg[p][f"{dim}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[p][f"{dim}_n"] = len(vals)
    return agg


def correlate_ragas_vs_real(
    agg: dict[str, dict[str, float]],
    real_target: dict[str, float],
    roster: list[str],
    dims: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Pearson r and Kendall tau between per-pipeline RAGAS means and a real-query target
    metric (e.g. mean nDCG@10), per dimension.
    """
    if dims is None:
        dims = ALL_DIMS

    real_vec = np.array([real_target[p] for p in roster])
    out: dict[str, dict[str, float]] = {}
    for dim in dims:
        ragas_vec = np.array([agg[p][f"{dim}_mean"] for p in roster])
        valid = ~np.isnan(ragas_vec) & ~np.isnan(real_vec)
        n = int(valid.sum())
        if n < 3:
            out[dim] = {"pearson_r": float("nan"), "kendall_tau": float("nan"), "n_valid": n}
            continue
        r = float(pearsonr(ragas_vec[valid], real_vec[valid])[0])
        t = float(kendalltau(ragas_vec[valid], real_vec[valid]).statistic)
        out[dim] = {"pearson_r": r, "kendall_tau": t, "n_valid": n}
    return out


def merge_dataset_results(
    dataset_results: list[dict],
    dims: list[str] | None = None,
) -> dict:
    """Combine per-dataset correlation results into one cross-dataset summary table."""
    if dims is None:
        dims = ALL_DIMS

    rows = []
    for d in dataset_results:
        corrs = d.get("correlations", {})
        row = {"dataset": d["dataset"]}
        for dim in dims:
            c = corrs.get(dim, {})
            row[f"{dim}_r"] = c.get("pearson_r", float("nan"))
            row[f"{dim}_tau"] = c.get("kendall_tau", float("nan"))
        rows.append(row)
    return {"datasets": rows, "dims": dims}
