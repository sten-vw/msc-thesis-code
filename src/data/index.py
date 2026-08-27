"""First-stage retrieval indexes with on-disk caches.

BM25 (Robertson & Zaragoza 2009), RM3 pseudo-relevance feedback (Lavrenko &
Croft 2001), SPLADE++ (Formal et al. 2022), single-vector dense retrieval, and
ColBERTv2 late interaction (Santhanam et al. 2022) via PyLate.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from core.types import Document

logger = logging.getLogger(__name__)

COLBERT_MODEL = "colbert-ir/colbertv2.0"
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


def _cache_root() -> str:
    from settings import CACHE_DIR

    return os.environ.get("RAG_EVAL_CACHE_DIR", str(CACHE_DIR))


def _dense_cache_dir() -> str:
    return f"{_cache_root()}/dense"


def _colbert_index_dir() -> str:
    return f"{_cache_root()}/colbert"


def _splade_cache_dir() -> str:
    return _cache_root()


def free_mem() -> None:
    """Release cached MPS/CUDA allocator buffers between heavy index builds."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _pick_index_device() -> str | None:
    import torch

    override = os.environ.get("RAG_EVAL_INDEX_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RetrievalIndex(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        ...


class BM25Index:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        from rank_bm25 import BM25Okapi

        self._doc_ids = [doc.doc_id for doc in documents]
        tokenized = [f"{doc.title} {doc.text}".lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        logger.info("Built BM25 index over %d documents", len(documents))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            (self._doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0
        ][:top_k]


class RM3BM25Index:
    """BM25 with RM3 pseudo-relevance feedback (Lavrenko & Croft 2001).

    First-pass BM25, RM1 relevance model over the top feedback documents,
    interpolation with the original query terms, then a per-term-weighted
    BM25 re-score.
    """

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
        fb_docs: int = 10,
        fb_terms: int = 10,
        original_weight: float = 0.5,
    ) -> None:
        from rank_bm25 import BM25Okapi

        self._doc_ids = [d.doc_id for d in documents]
        self._toks = [f"{d.title} {d.text}".lower().split() for d in documents]
        self._bm25 = BM25Okapi(self._toks, k1=k1, b=b)
        self._fb_docs = fb_docs
        self._fb_terms = fb_terms
        self._orig_weight = original_weight
        logger.info("Built RM3+BM25 index over %d documents", len(documents))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        from collections import Counter

        import numpy as np

        qtok = query.lower().split()
        scores = self._bm25.get_scores(qtok)
        top = scores.argsort()[-self._fb_docs:][::-1]
        rm: Counter = Counter()
        for di in top:
            if scores[di] <= 0:
                continue
            doc = self._toks[di]
            if not doc:
                continue
            tf = Counter(doc)
            inv = 1.0 / len(doc)
            for t, c in tf.items():
                rm[t] += c * inv
        fb = [t for t, _ in rm.most_common(self._fb_terms)]
        weights: Counter = Counter()
        for t in qtok:
            weights[t] += self._orig_weight / max(1, len(qtok))
        tot = sum(rm[t] for t in fb) or 1.0
        for t in fb:
            weights[t] += (1 - self._orig_weight) * rm[t] / tot
        agg = np.zeros(len(self._doc_ids))
        for t, w in weights.items():
            agg += w * self._bm25.get_scores([t])
        topk = agg.argsort()[-top_k:][::-1]
        return [(self._doc_ids[i], float(agg[i])) for i in topk if agg[i] > 0]


class SpladeIndex:
    """SPLADE++ learned sparse retrieval (Formal et al. 2021, 2022); document matrix held as a
    scipy CSR, cached per corpus.
    """

    def __init__(
        self,
        documents: list[Document],
        model_name: str = SPLADE_MODEL,
        max_length: int = 256,
        batch_size: int = 16,
        cache_key: str | None = None,
    ) -> None:
        import numpy as np
        import scipy.sparse as sp
        import torch
        from tqdm import tqdm
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._doc_ids = [doc.doc_id for doc in documents]
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._model.eval()
        self._device = torch.device(_pick_index_device() or "cpu")
        self._model.to(self._device)
        self._max_length = max_length
        self._vocab_size = self._model.config.vocab_size

        cache_path = (
            Path(_splade_cache_dir()) / f"splade_{cache_key}.npz" if cache_key else None
        )
        if cache_path and cache_path.exists():
            cached = sp.load_npz(cache_path).tocsr()
            if cached.shape == (len(documents), self._vocab_size):
                self._matrix = cached
                logger.info("Reusing cached SPLADE matrix %s", cache_path.name)
                return
            logger.warning("Ignoring stale SPLADE cache %s; re-encoding.", cache_path.name)

        texts = [f"{doc.title} {doc.text}".strip() for doc in documents]
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        with torch.no_grad():
            for batch_start in tqdm(
                range(0, len(texts), batch_size),
                desc=f"SPLADE encode ({len(texts)} docs)",
                unit="batch",
            ):
                batch = texts[batch_start : batch_start + batch_size]
                tok = self._tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=max_length, return_tensors="pt",
                ).to(self._device)
                logits = self._model(**tok).logits
                mask = tok["attention_mask"].unsqueeze(-1)
                scores = torch.log1p(torch.relu(logits)) * mask
                doc_vecs = scores.max(dim=1).values.cpu().numpy()
                for i, vec in enumerate(doc_vecs):
                    nz = np.nonzero(vec)[0]
                    if nz.size == 0:
                        continue
                    rows.extend([batch_start + i] * nz.size)
                    cols.extend(nz.tolist())
                    vals.extend(vec[nz].tolist())

        self._matrix = sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(texts), self._vocab_size), dtype=np.float32,
        )
        logger.info("Built SPLADE index over %d documents", len(texts))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
            sp.save_npz(tmp, self._matrix)
            os.replace(tmp, cache_path)

    def _encode_query(self, query: str):
        import numpy as np
        import torch

        with torch.no_grad():
            tok = self._tokenizer(
                [query], truncation=True, padding=True,
                max_length=self._max_length, return_tensors="pt",
            ).to(self._device)
            logits = self._model(**tok).logits
            mask = tok["attention_mask"].unsqueeze(-1)
            scores = torch.log1p(torch.relu(logits)) * mask
            return scores.max(dim=1).values.cpu().numpy().astype(np.float32)[0]

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        import numpy as np

        scores = self._matrix @ self._encode_query(query)
        if scores.size == 0:
            return []
        top_idx = np.argpartition(-scores, min(top_k, scores.size - 1))[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._doc_ids[int(i)], float(scores[i])) for i in top_idx if scores[i] > 0]


class DenseIndex:
    """Single-vector dense retrieval over a FAISS flat index, cached per corpus and model."""

    def __init__(
        self,
        documents: list[Document],
        model_name: str,
        similarity: str = "cosine",
        query_prefix: str = "",
        doc_prefix: str = "",
        dataset: str | None = None,
    ) -> None:
        import json

        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass

        if similarity not in ("cosine", "dot"):
            raise ValueError(f"similarity must be 'cosine' or 'dot', got {similarity!r}")

        self._doc_ids = [doc.doc_id for doc in documents]
        self._similarity = similarity
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix

        slug = model_name.replace("/", "_")
        cache_index_path = cache_ids_path = None
        if dataset:
            cache_dir = Path(_dense_cache_dir())
            cache_index_path = cache_dir / f"dense_{dataset}_{slug}.faiss"
            cache_ids_path = cache_dir / f"dense_{dataset}_{slug}.ids.json"
            if cache_index_path.exists() and cache_ids_path.exists():
                cached_ids = json.loads(cache_ids_path.read_text())
                if len(cached_ids) == len(documents):
                    self._index = faiss.read_index(str(cache_index_path))
                    self._doc_ids = cached_ids
                    self._model = SentenceTransformer(model_name, device="cpu")
                    logger.info("Reusing cached dense index %s", cache_index_path)
                    return
                logger.warning("Dense cache %s stale; rebuilding.", cache_index_path)

        device = (
            os.environ.get("RAG_EVAL_INDEX_DEVICE")
            or ("cpu" if "large" in model_name.lower() else None)
        )
        free_mem()
        model = SentenceTransformer(model_name, device=device)

        texts = [f"{doc_prefix}{doc.title} {doc.text}".strip() for doc in documents]
        embeddings = model.encode(
            texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)
        if similarity == "cosine":
            faiss.normalize_L2(embeddings)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._model = model
        logger.info("Built dense index over %d documents (%s)", len(documents), model_name)

        if cache_index_path is not None:
            cache_index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_idx = cache_index_path.with_suffix(f".{os.getpid()}.tmp")
            faiss.write_index(self._index, str(tmp_idx))
            os.replace(tmp_idx, cache_index_path)
            tmp_ids = cache_ids_path.with_suffix(f".{os.getpid()}.tmp")
            tmp_ids.write_text(json.dumps(self._doc_ids))
            os.replace(tmp_ids, cache_ids_path)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        import faiss
        import numpy as np

        q_emb = self._model.encode(
            [f"{self._query_prefix}{query}"], convert_to_numpy=True,
        ).astype(np.float32)
        if self._similarity == "cosine":
            faiss.normalize_L2(q_emb)
        scores, indices = self._index.search(q_emb, top_k)
        return [
            (self._doc_ids[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]


class ColBERTIndex:
    """ColBERTv2 late interaction (Santhanam et al. 2022) over a PyLate Voyager index."""

    def __init__(
        self, documents: list[Document], dataset: str, model_name: str = COLBERT_MODEL,
    ) -> None:
        import pickle

        try:
            from pylate import indexes, models, retrieve
        except ImportError as e:
            raise ImportError(
                "ColBERTIndex requires `pylate`. Install with `uv pip install pylate`."
            ) from e

        self._doc_ids = [doc.doc_id for doc in documents]
        device = os.environ.get("RAG_EVAL_INDEX_DEVICE") or "cpu"
        batch_size = 32
        if len(documents) > 50_000:
            device = "cpu"
            batch_size = 8

        self._model = models.ColBERT(model_name_or_path=model_name, device=device)

        index_dir = Path(_colbert_index_dir()) / dataset
        index_path = index_dir / "index.voyager"
        ids_map_path = index_dir / "document_ids_to_embeddings.pkl"
        reuse = False
        if index_path.exists() and ids_map_path.exists():
            try:
                with open(ids_map_path, "rb") as f:
                    cached_n = len(pickle.load(f))
                reuse = cached_n == len(documents)
            except Exception:
                logger.warning("ColBERT cache %s unreadable; rebuilding.", index_dir)

        if reuse:
            self._index = indexes.Voyager(
                index_folder=_colbert_index_dir(), index_name=dataset, override=False,
            )
            logger.info("Reusing cached ColBERT index %s", index_dir)
        else:
            self._index = indexes.Voyager(
                index_folder=_colbert_index_dir(), index_name=dataset, override=True,
            )
            docs_emb = self._model.encode(
                [f"{doc.title} {doc.text}".strip() for doc in documents],
                batch_size=batch_size, is_query=False, show_progress_bar=True,
            )
            self._index.add_documents(
                documents_ids=self._doc_ids, documents_embeddings=docs_emb,
            )
            logger.info("Built ColBERT index over %d documents", len(documents))

        self._retr = retrieve.ColBERT(index=self._index)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = self._model.encode([query], is_query=True, show_progress_bar=False)
        res = self._retr.retrieve(queries_embeddings=q_emb, k=top_k)[0]
        return [(r["id"], float(r["score"])) for r in res]
