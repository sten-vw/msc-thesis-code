"""The frozen eight-pipeline retrieval set: build indexes, run a query pool, assemble runs.

Direct first stages, reciprocal-rank fusions (Cormack et al. 2009) and a rerank
cascade. Index construction and its disk caches live in ``data.index``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from core.types import Document, Query
from data.index import (
    BM25Index,
    ColBERTIndex,
    DenseIndex,
    RM3BM25Index,
    SpladeIndex,
    free_mem,
)
from settings import POOL_DEPTH

CACHE_DEPTH = 100

DENSE_MODELS = {
    "d_gte_base": ("thenlper/gte-base", "", "", "cosine"),
    "d_e5_base": ("intfloat/e5-base-v2", "query: ", "passage: ", "cosine"),
}

FUSION_MEMBERS = {
    "rrf_bm25_splade": ["bm25", "splade"],
    "rrf_splade_d_e5_base": ["splade", "d_e5_base"],
}

CASCADES = {"bm25__ce": ("bm25", 50)}


def needed_first_stages(roster: list[str]) -> set[str]:
    need: set[str] = set()
    for p in roster:
        if p in FUSION_MEMBERS:
            need.update(FUSION_MEMBERS[p])
        elif p in CASCADES:
            need.add(CASCADES[p][0])
        else:
            need.add(p)
    return need


def build_first_stages(need: set[str], docs: list[Document], dataset: str) -> dict:
    idx: dict = {}

    def add(name: str, builder) -> None:
        if name not in need:
            return
        t0 = time.time()
        idx[name] = builder()
        print(f"  built {name:<16} [{time.time() - t0:.0f}s]", flush=True)
        free_mem()

    add("bm25", lambda: BM25Index(docs))
    add("rm3_bm25", lambda: RM3BM25Index(docs))
    add("splade", lambda: SpladeIndex(docs, cache_key=dataset))
    for label in [n for n in need if n.startswith("d_")]:
        model_id, query_prefix, doc_prefix, similarity = DENSE_MODELS[label]
        idx[label] = DenseIndex(
            docs, model_name=model_id, similarity=similarity,
            query_prefix=query_prefix, doc_prefix=doc_prefix, dataset=dataset,
        )
        free_mem()
        print(f"  built {label}", flush=True)
    if "colbertv2" in need:
        idx["colbertv2"] = ColBERTIndex(docs, dataset)
        free_mem()
        print("  built colbertv2", flush=True)
    return idx


def run_first_stages(idx: dict, queries: dict[str, Query]) -> dict:
    runs: dict = {}
    for name, index in idx.items():
        t0 = time.time()
        runs[name] = {q: index.search(queries[q].text, top_k=CACHE_DEPTH) for q in queries}
        print(
            f"    ran {name:<16} over {len(queries)} q [{time.time() - t0:.0f}s]",
            flush=True,
        )
    return runs


def rrf_fuse(member_runs: list[dict], qids, rrf_k: int = 60, pool: int = 40,
             depth: int = CACHE_DEPTH) -> dict:
    """Reciprocal-rank fusion over precomputed first-stage runs (Cormack et al. 2009)."""
    out: dict = {}
    for q in qids:
        scores: dict[str, float] = {}
        for run in member_runs:
            for rank, (did, _) in enumerate(run[q][:pool]):
                scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        out[q] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:depth]
    return out


def assemble_roster_runs(roster: list[str], fs_runs: dict,
                         queries: dict[str, Query], corpus: dict) -> dict:
    """{pipeline: {qid: [(doc_id, score), ...]}} for every pipeline in the set."""
    from retrieval.reranker import CrossEncoderReranker

    qids = list(queries)
    runs: dict = {}
    reranker = None
    for p in roster:
        if p in FUSION_MEMBERS:
            runs[p] = rrf_fuse([fs_runs[m] for m in FUSION_MEMBERS[p] if m in fs_runs], qids)
        elif p in CASCADES:
            base, pool = CASCADES[p]
            if reranker is None:
                reranker = CrossEncoderReranker()
            runs[p] = {
                q: (
                    reranker.rerank(
                        queries[q].text,
                        [d for d, _ in fs_runs[base][q][:pool]],
                        corpus, top_k=CACHE_DEPTH,
                    )
                    if fs_runs[base][q] else []
                )
                for q in qids
            }
        else:
            runs[p] = fs_runs[p]
    return runs


def runs_to_rows(runs: dict, roster: list[str]) -> list[dict]:
    return [
        {"retriever": p, "query_id": qid, "doc_id": did, "rank": rank, "score": float(sc)}
        for p in roster
        for qid, docs in runs[p].items()
        for rank, (did, sc) in enumerate(docs)
    ]


def pool_from_runs(runs, roster: list[str] | None = None, *, depth: int = POOL_DEPTH,
                   source_docs: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Depth-K cross-pipeline judgment pool {qid: [doc_ids]}, unioning source docs."""
    pool: dict[str, set] = {}
    if isinstance(runs, pd.DataFrame):
        sub = runs[runs["rank"] < depth]
        for qid, did in zip(sub.query_id, sub.doc_id):
            pool.setdefault(qid, set()).add(did)
    else:
        for p in (roster if roster is not None else list(runs)):
            for qid, docs in runs[p].items():
                for did, _ in docs[:depth]:
                    pool.setdefault(qid, set()).add(did)
    if source_docs:
        for qid, src in source_docs.items():
            if src and qid in pool:
                pool[qid].add(src)
    return {q: sorted(d) for q, d in pool.items()}


def save_runs(runs: dict, roster: list[str], runs_path: str | Path) -> None:
    runs_path = Path(runs_path)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    rows = runs_to_rows(runs, roster)
    pd.DataFrame(rows).to_parquet(runs_path)
    print(f"  [saved {runs_path.name}: {len(rows)} run rows]", flush=True)


def save_pool(pool: dict[str, list[str]], pool_path: str | Path) -> None:
    pool_path = Path(pool_path)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(pool))


def retrieve_pool(
    query_texts: dict[str, str],
    source_docs: dict[str, str],
    corpus: dict,
    idx: dict,
    roster: list[str],
) -> dict:
    """Assemble roster runs for an in-memory query pool over prebuilt indexes."""
    queries = {
        qid: Query(query_id=qid, text=text or "(empty)", source_doc_id=source_docs.get(qid))
        for qid, text in query_texts.items()
    }
    fs = run_first_stages(idx, queries)
    return assemble_roster_runs(roster, fs, queries, corpus)
