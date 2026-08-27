"""Cross-encoder reranker for the rerank cascade (Nogueira et al. 2020)."""

from __future__ import annotations

from core.types import Document


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        doc_ids: list[str],
        corpus: dict[str, Document],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        pairs = []
        valid_ids = []
        for did in doc_ids:
            doc = corpus.get(did)
            if doc:
                pairs.append((query, f"{doc.title} {doc.text}"))
                valid_ids.append(did)
        if not pairs:
            return []

        scores = self._get_model().predict(pairs)
        scored = sorted(
            zip(valid_ids, [float(s) for s in scores]), key=lambda x: x[1], reverse=True
        )
        return scored[:top_k] if top_k else scored
