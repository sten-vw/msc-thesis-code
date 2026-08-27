"""Naive zero-shot doc->query generation.

The floor arm of the generator ladder: no few-shot demonstrations and no
query-shaping system persona, just the document and a single instruction
to write a query for it (samples with ``system=None``). One query per
document, tagged ``generation_strategy="naive"``.
"""

from __future__ import annotations

import logging

from tqdm import tqdm

import settings
from core.llm import BedrockLLM
from core.types import Document, Query
from generation.base import make_query
from generation.parsing import first_query

logger = logging.getLogger(__name__)

_MAX_NEW_TOKENS = 512


def _build_prompt(document: Document) -> str:
    return (
        f"Document:\n{document.text}\n\n"
        "Write a query that this document answers. Output only the query."
    )


class NaiveGenerator:
    """One query per document from a single unopinionated instruction: no exemplars, no system
    persona.
    """

    def __init__(
        self, llm: BedrockLLM, *, temperature: float = settings.GENERATION_TEMPERATURE,
    ) -> None:
        self.llm = llm
        self.temperature = temperature

    def generate(self, documents: list[Document]) -> list[Query]:
        prompts = [_build_prompt(doc) for doc in documents]
        message_batches = [[BedrockLLM.user_message(p)] for p in prompts]

        pbar = tqdm(total=len(documents), desc="naive generation")

        def on_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        responses = self.llm.invoke_batch(
            message_batches,
            system=None,
            temperature=self.temperature,
            max_tokens=_MAX_NEW_TOKENS,
            on_progress=on_progress,
        )
        pbar.close()

        all_queries: list[Query] = []
        for doc, response in zip(documents, responses):
            text = first_query(response.text)
            if text:
                all_queries.append(make_query(text, doc, "naive"))

        logger.info(
            "naive: generated %d queries from %d documents", len(all_queries), len(documents)
        )
        return all_queries
