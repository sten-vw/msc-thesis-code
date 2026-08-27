"""Cluster-then-generate (DUQGen).

Faithful reimplementation of DUQGen's diversity-aware document selection:
K-Means over document embeddings -> stratified per-cluster allocation
proportional to cluster size -> centroid-weighted softmax sampling across
5 fixed seed rounds (pooled) -> MMR selection of the per-cluster set ->
1-shot in-domain query generation from each selected representative doc.

Source: DUQGen (Chandradevan et al., NAACL 2024)
  Paper: https://aclanthology.org/2024.naacl-long.413/
  Repo:  https://github.com/emory-irlab/DUQGen
"""

from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

import settings
from core.llm import BedrockLLM
from core.types import Document, Query
from generation.parsing import parse_queries
from generation.prompts import QUERY_GEN_SYSTEM

logger = logging.getLogger(__name__)

# DUQGen sample_target_collection_documents.py: SEED_LIST = [35, 745, 10, 6534, 2].
DUQGEN_SEED_LIST: tuple[int, ...] = (35, 745, 10, 6534, 2)


def _build_prompt(
    document: Document,
    num_queries: int,
    example_query: str | None,
) -> str:
    """DUQGen in-domain prompt: an optional example query, then the target document with an
    empty query to complete.
    """
    parts: list[str] = []

    if example_query:
        parts.append("Here is an example of a relevant query from this domain:")
        parts.append(f"Relevant Query: {example_query}")
        parts.append("")

    if num_queries > 1:
        parts.append(
            f"Generate {num_queries} relevant search queries that the "
            f"following document can answer. Output one query per line and "
            f"nothing else."
        )
    else:
        parts.append(
            "Generate one relevant search query that the following document "
            "can answer. Output only the query and nothing else."
        )
    parts.append("")
    parts.append(f"Document: {document.text}")
    parts.append("Relevant Query:")

    return "\n".join(parts)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = values / max(temperature, 1e-8)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _allocation_size(
    cluster_size: int,
    n_clusters: int,
    n_corpus: int,
    budget: int,
) -> int:
    """DUQGen repo formula: sample_size = 1 + floor((n_train - n_clusters) * size / n_corpus),
    n_train=budget.
    """
    remaining = max(budget - n_clusters, 0)
    return 1 + int(math.floor(remaining * (cluster_size / max(n_corpus, 1))))


def _pool_candidates(
    similarities: np.ndarray,
    sample_size: int,
    temperature: float,
    seeds: tuple[int, ...],
) -> list[int]:
    """Union of softmax draws across the fixed DUQGen seed rounds, without replacement per
    round.
    """
    n = len(similarities)
    if n <= sample_size:
        return list(range(n))

    probs = _softmax(similarities, temperature)
    pool: set[int] = set()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        picks = rng.choice(n, size=sample_size, replace=False, p=probs)
        pool.update(int(p) for p in picks)
    return sorted(pool)


def _mmr_select(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    k: int,
    lam: float,
) -> list[int]:
    """MMR selection: greedily maximises lam*sim_to_centroid - (1-lam)*max sim to selected,
    until k chosen.
    """
    n = len(embeddings)
    if n <= k:
        return list(range(n))

    relevance = _cosine_similarity(centroid, embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    unit = embeddings / norms
    pairwise = unit @ unit.T

    selected: list[int] = []
    remaining = set(range(n))
    for _ in range(k):
        best_score = -np.inf
        best_idx = -1
        for idx in remaining:
            if selected:
                sim_to_selected = max(float(pairwise[idx, s]) for s in selected)
            else:
                sim_to_selected = 0.0
            score = lam * float(relevance[idx]) - (1.0 - lam) * sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


class DuqgenGenerator:
    """Cluster documents, then generate from representative docs per cluster."""

    def __init__(
        self,
        llm: BedrockLLM,
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
        temperature: float = settings.GENERATION_TEMPERATURE,
        seed: int = settings.SPLIT_SEED,
        num_clusters: int = 1000,
        generation_budget: int | None = None,
        sampling_temperature: float = 1.0,
        mmr_lambda: float = 1.0,
    ) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.seed = seed
        self.num_clusters = num_clusters
        self.generation_budget = generation_budget
        self.sampling_temperature = sampling_temperature
        self.mmr_lambda = mmr_lambda

    def generate(
        self,
        documents: list[Document],
        *,
        few_shot_queries: list[Query] | None = None,
        num_queries: int | None = None,
        num_queries_per_doc: int = 1,
    ) -> list[Query]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedding_model)

        if num_queries:
            per_doc = max(num_queries_per_doc, 1)
            budget = math.ceil(num_queries / per_doc)
        else:
            budget = self.generation_budget or self.num_clusters

        n_clusters = min(self.num_clusters, len(documents), budget)
        n_clusters = max(n_clusters, 1)

        example_texts: list[str] = []
        if few_shot_queries:
            example_texts = [q.text for q in few_shot_queries if q.text]
        if not example_texts:
            logger.warning(
                "DUQGen: no in-domain example queries supplied "
                "(few_shot_queries empty/None); generating without the "
                "in-domain prompt."
            )

        logger.info("DUQGen: embedding %d documents for clustering...", len(documents))
        texts = [f"{doc.title} {doc.text}".strip() for doc in documents]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        logger.info("DUQGen: clustering into %d clusters...", n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        n_corpus = int(cluster_sizes.sum())
        docs_per_cluster = np.array(
            [
                _allocation_size(int(sz), n_clusters, n_corpus, budget)
                for sz in cluster_sizes
            ],
            dtype=int,
        )

        docs_per_cluster = np.minimum(docs_per_cluster, cluster_sizes)
        while docs_per_cluster.sum() > budget:
            shrinkable = np.where(docs_per_cluster > 1)[0]
            if len(shrinkable) == 0:
                break
            share = cluster_sizes[shrinkable] / max(n_corpus, 1) * budget
            excess = docs_per_cluster[shrinkable] - share
            docs_per_cluster[shrinkable[int(np.argmax(excess))]] -= 1
        while docs_per_cluster.sum() < budget:
            growable = np.where(docs_per_cluster < cluster_sizes)[0]
            if len(growable) == 0:
                break
            share = cluster_sizes[growable] / max(n_corpus, 1) * budget
            deficit = share - docs_per_cluster[growable]
            docs_per_cluster[growable[int(np.argmax(deficit))]] += 1

        representative_docs: list[Document] = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_embeddings = embeddings[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]
            k = int(docs_per_cluster[cluster_id])
            if k <= 0:
                continue

            if len(cluster_indices) <= k:
                selected_global = cluster_indices.tolist()
            else:
                sims = _cosine_similarity(centroid, cluster_embeddings)
                candidate_local = _pool_candidates(
                    sims, k, self.sampling_temperature, DUQGEN_SEED_LIST,
                )
                candidate_embeddings = cluster_embeddings[candidate_local]
                mmr_indices = _mmr_select(
                    candidate_embeddings, centroid, k, self.mmr_lambda,
                )
                selected_global = [
                    int(cluster_indices[candidate_local[i]]) for i in mmr_indices
                ]

            for g_idx in selected_global:
                representative_docs.append(documents[g_idx])

        logger.info(
            "DUQGen: selected %d representative docs across %d clusters (budget=%d)",
            len(representative_docs), n_clusters, budget,
        )

        message_batches = []
        for i, doc in enumerate(representative_docs):
            example = example_texts[i % len(example_texts)] if example_texts else None
            message_batches.append(
                [BedrockLLM.user_message(_build_prompt(doc, num_queries_per_doc, example))]
            )

        pbar = tqdm(total=len(representative_docs), desc="DUQGen cluster generation")

        def on_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        responses = self.llm.invoke_batch(
            message_batches,
            system=QUERY_GEN_SYSTEM,
            temperature=self.temperature,
            on_progress=on_progress,
        )
        pbar.close()

        all_queries: list[Query] = []
        for doc, response in zip(representative_docs, responses):
            queries = parse_queries(
                response.text, doc, "duqgen", max_queries=num_queries_per_doc,
            )
            all_queries.extend(queries)

        logger.info(
            "DUQGen: generated %d queries from %d representative docs (%d clusters)",
            len(all_queries), len(representative_docs), n_clusters,
        )
        return all_queries
