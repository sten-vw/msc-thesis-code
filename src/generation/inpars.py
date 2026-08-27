"""InPars vanilla doc->query generation.

Faithful reimplementation of the InPars and InPars-v2 *generation* step:
GBQ 3-shot (document, good question, bad question) MS MARCO exemplars,
then the target document ending on a dangling "Good Question:" line. A
vanilla 3-shot template without the bad-question contrast is available via
``gbq=False``. Filtering (log-prob / monoT5 rerank) is a separate strategy,
out of scope here.

The exemplars are the fixed MS MARCO ones shipped in the InPars repo, not
in-domain pairs from the target corpus. Documents are passed whole, dropping
InPars's 256-token document and 64-token generation caps, which existed for
GPT-J's 2048-token context.

Source: InPars (Bonifacio et al., SIGIR 2022)
  Paper: arXiv:2202.05144
Source: InPars-v2 (Jeronymo et al., 2023)
  Paper: arXiv:2301.01820
"""

from __future__ import annotations

import logging

from tqdm import tqdm

import settings
from core.llm import BedrockLLM
from core.types import Document, Query
from generation.base import make_query
from generation.parsing import first_query
from generation.prompts import QUERY_GEN_SYSTEM

logger = logging.getLogger(__name__)

_MAX_NEW_TOKENS = 512

# Exact few-shot triples from the InPars `inpars-gbq` template.
# https://github.com/zetaalphavector/InPars/blob/master/inpars/prompts/templates.yaml
FEW_SHOT_EXAMPLES = [
    {
        "document": (
            "We don't know a lot about the effects of caffeine during pregnancy "
            "on you and your baby. So it's best to limit the amount you get each "
            "day. If you are pregnant, limit caffeine to 200 milligrams each day. "
            "This is about the amount in 1½ 8-ounce cups of coffee or one "
            "12-ounce cup of coffee."
        ),
        "good_query": "How much caffeine is ok for a pregnant woman to have?",
        "bad_query": "Is a little caffeine ok during pregnancy?",
    },
    {
        "document": (
            "Passiflora herbertiana. A rare passion fruit native to Australia. "
            "Fruits are green-skinned, white fleshed, with an unknown edible "
            "rating. Some sources list the fruit as edible, sweet and tasty, "
            "while others list the fruits as being bitter and inedible."
        ),
        "good_query": (
            "What is Passiflora herbertiana (a rare passion fruit) and "
            "how does it taste like?"
        ),
        "bad_query": "What fruit is native to Australia?",
    },
    {
        "document": (
            "The Canadian Armed Forces. 1  The first large-scale Canadian "
            "peacekeeping mission started in Egypt on November 24, 1956. "
            "2  There are approximately 65,000 Regular Force and 25,000 "
            "reservist members in the Canadian military. 3  In Canada, "
            "August 9 is designated as National Peacekeepers' Day."
        ),
        "good_query": (
            "Information on the Canadian Armed Forces size and history."
        ),
        "bad_query": "How large is the Canadian military?",
    },
]


def _build_gbq_prompt(document: Document) -> str:
    """The `inpars-gbq` prompt, ending with a dangling ``Good Question:`` line."""
    parts: list[str] = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Document: {ex['document']}")
        parts.append(f"Good Question: {ex['good_query']}")
        parts.append(f"Bad Question: {ex['bad_query']}")
        parts.append("")

    parts.append(f"Example {len(FEW_SHOT_EXAMPLES) + 1}:")
    parts.append(f"Document: {document.text}")
    parts.append("Good Question:")
    return "\n".join(parts)


def _build_vanilla_prompt(document: Document) -> str:
    """The vanilla `inpars` prompt, ending with a dangling ``Relevant Query:`` line."""
    parts: list[str] = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Document: {ex['document']}")
        parts.append(f"Relevant Query: {ex['bad_query']}")
        parts.append("")

    parts.append(f"Example {len(FEW_SHOT_EXAMPLES) + 1}:")
    parts.append(f"Document: {document.text}")
    parts.append("Relevant Query:")
    return "\n".join(parts)


class InParsGenerator:
    """InPars generation: one query per document via the GBQ (or vanilla) few-shot template."""

    def __init__(
        self,
        llm: BedrockLLM,
        *,
        temperature: float = settings.GENERATION_TEMPERATURE,
        gbq: bool = True,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.gbq = gbq

    def _build_prompt(self, document: Document) -> str:
        return _build_gbq_prompt(document) if self.gbq else _build_vanilla_prompt(document)

    def generate(self, documents: list[Document]) -> list[Query]:
        prompts = [self._build_prompt(doc) for doc in documents]
        message_batches = [[BedrockLLM.user_message(p)] for p in prompts]

        pbar = tqdm(total=len(documents), desc="InPars generation")

        def on_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        responses = self.llm.invoke_batch(
            message_batches,
            system=QUERY_GEN_SYSTEM,
            temperature=self.temperature,
            max_tokens=_MAX_NEW_TOKENS,
            on_progress=on_progress,
        )
        pbar.close()

        all_queries: list[Query] = []
        for doc, response in zip(documents, responses):
            text = first_query(response.text)
            if text:
                all_queries.append(make_query(text, doc, "inpars"))

        logger.info(
            "InPars: generated %d queries from %d documents", len(all_queries), len(documents)
        )
        return all_queries
