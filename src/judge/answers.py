"""Answer generation from retrieved context, with a resume-safe batch cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from core.llm import BedrockLLM
from core.types import Document

GENERATION_SYSTEM = """\
You are a helpful assistant. Answer the user's question based on the provided \
context documents. If the context doesn't contain enough information to answer, \
say so. Be concise and accurate."""

_GENERATION_PROMPT = """\
Context:
{context}

Question: {question}

Answer:"""

_PER_DOC_CHARS = 16_000


def build_context(
    retrieved_docs: list[tuple[str, float]],
    corpus: dict[str, Document],
    top_k: int = 5,
) -> str:
    """Assemble context string from the top-k retrieved docs."""
    parts = []
    for doc_id, _score in retrieved_docs[:top_k]:
        doc = corpus.get(doc_id)
        if doc:
            text = f"{doc.title}\n{doc.text}" if doc.title else doc.text
            parts.append(text[:_PER_DOC_CHARS])
    return "\n\n".join(parts) if parts else "No relevant documents found."


class AnswerGenerator:
    """Generate answers from retrieved context using an LLM."""

    def __init__(self, llm: BedrockLLM, top_k: int = 5) -> None:
        self.llm = llm
        self.top_k = top_k

    def generate(
        self,
        query: str,
        retrieved_docs: list[tuple[str, float]],
        corpus: dict[str, Document],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """`system` overrides the default generation prompt; `temperature` is forwarded to the
        LLM.
        """
        context = build_context(retrieved_docs, corpus, self.top_k)
        prompt = _GENERATION_PROMPT.format(context=context, question=query)
        response = self.llm.invoke(
            [BedrockLLM.user_message(prompt)],
            system=system or GENERATION_SYSTEM,
            temperature=temperature,
        )
        return response.text.strip()

    def generate_batch_cached(
        self,
        queries: dict[str, str],
        retrieval: dict[str, dict[str, list[tuple[str, float]]]],
        corpus: dict[str, Document],
        pipelines: list[str],
        cache_path: Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, dict[str, str]]:
        """JSONL cache, one {"pipeline", "query_id", "answer"} object per line; only pairs
        missing from it are generated.
        """
        cache: dict[str, dict[str, str]] = {}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    obj = json.loads(line)
                    cache.setdefault(obj["pipeline"], {})[obj["query_id"]] = obj["answer"]

        needed = [
            (p, q)
            for p in pipelines
            for q in queries
            if q not in cache.get(p, {})
        ]

        done = 0
        with open(cache_path, "a") as fh:
            for pipeline, qid in needed:
                docs = sorted(
                    retrieval.get(pipeline, {}).get(qid, []), key=lambda x: -x[1]
                )
                answer = self.generate(queries[qid], docs, corpus)
                obj = {"pipeline": pipeline, "query_id": qid, "answer": answer}
                fh.write(json.dumps(obj) + "\n")
                cache.setdefault(pipeline, {})[qid] = answer
                done += 1
                if on_progress:
                    on_progress(done, len(needed))

        return cache


def load_answer_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    """Replay a `generate_batch_cached` JSONL cache without re-generating."""
    out: dict[str, dict[str, str]] = {}
    if not Path(cache_path).exists():
        return out
    for line in Path(cache_path).read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            out.setdefault(obj["pipeline"], {})[obj["query_id"]] = obj["answer"]
    return out
