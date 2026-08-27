"""UDAPDR two-stage seed+scale synthetic query generation.

Faithful reimplementation of UDAPDR Stages 1-3 (Saad-Falcon et al., EMNLP
2023, arXiv:2303.00807), fully unsupervised: Stage 1 runs a strong seed model
over 5 zero-/few-shot prompts on X sampled passages to build (good, bad)
demonstration pairs; Stage 2 assembles Y corpus-adapted prompt variants from
those pairs; Stage 3 applies each variant with a cheap scale model across the
corpus, one query per passage, with `source_doc_id` as the relevance label.
The seed and scale models are the Bedrock ones in `settings`, not the paper's
GPT-3 and Flan-T5-XXL.

Sources:
  * Stage-1 prompts: Figure 2 (transcribed from the figure image at
    ar5iv.labs.arxiv.org/html/2303.00807).
  * Stage-2 template: Figure 3.
  * InPars few-shot examples: Bonifacio et al. 2022,
    https://github.com/zetaalphavector/InPars (gbq_prompt / vanilla_prompt).
"""

from __future__ import annotations

import logging
import random

from tqdm import tqdm

import settings
from core.llm import BedrockLLM
from core.types import Document, Query
from generation.base import make_query
from generation.parsing import clean_query_line, first_query
from generation.prompts import QUERY_GEN_SYSTEM

logger = logging.getLogger(__name__)

# UDAPDR §3: X in {5, 10, 50, 100}; 50 is the paper's modal value.
DEFAULT_SEED_PASSAGES = 50

# UDAPDR Figure 3: 3 demonstration triples per Stage-2 prompt.
DEMOS_PER_VARIANT = 3

# UDAPDR §3: Y in {1, 5, 10}; main results use Y=5.
DEFAULT_NUM_VARIANTS = 5

# InPars few-shot examples (Bonifacio et al. 2022), reproduced in UDAPDR Figure 2.
_INPARS_EXAMPLES = [
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
            "The Canadian Armed Forces. 1 The first large-scale Canadian "
            "peacekeeping mission started in Egypt on November 24, 1956. "
            "2 There are approximately 65,000 Regular Force and 25,000 "
            "reservist members in the Canadian military. 3 In Canada, "
            "August 9 is designated as National Peacekeepers' Day."
        ),
        "good_query": "Information on the Canadian Armed Forces size and history",
        "bad_query": "How large is the Canadian military?",
    },
]


def _gbq_examples_block() -> list[str]:
    parts: list[str] = []
    for i, ex in enumerate(_INPARS_EXAMPLES, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Document: {ex['document']}")
        parts.append(f"Good Question: {ex['good_query']}")
        parts.append(f"Bad Question: {ex['bad_query']}")
        parts.append("")
    return parts


def _build_stage1_gbq(document: Document) -> str:
    """Prompt #1: InPars GBQ -- few-shot, dangling Good Question."""
    parts = _gbq_examples_block()
    parts.append(f"Example {len(_INPARS_EXAMPLES) + 1}:")
    parts.append(f"Document: {document.text}")
    parts.append("Good Question:")
    return "\n".join(parts)


def _build_stage1_vanilla(document: Document) -> str:
    """Prompt #2: InPars vanilla -- few-shot, single Relevant Query."""
    parts: list[str] = []
    for i, ex in enumerate(_INPARS_EXAMPLES, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Document: {ex['document']}")
        parts.append(f"Relevant Query: {ex['bad_query']}")
        parts.append("")
    parts.append(f"Example {len(_INPARS_EXAMPLES) + 1}:")
    parts.append(f"Document: {document.text}")
    parts.append("Relevant Query:")
    return "\n".join(parts)


def _build_stage1_zero1(document: Document) -> str:
    """Prompt #3: zero-shot -- Retrieve a Query."""
    return (
        "Retrieve a Query answered by the following Document.\n"
        f"Document: {document.text}\n"
        "Query:"
    )


def _build_stage1_zero2(document: Document) -> str:
    """Prompt #4: zero-shot -- Design a Question."""
    return (
        "Design a Question that is answered by the following Passage.\n"
        f"Passage: {document.text}\n"
        "Question:"
    )


def _build_stage1_zero3(document: Document) -> str:
    """Prompt #5: zero-shot -- Write a Question."""
    return (
        "Write a Question answered by the given Passage.\n"
        f"Passage: {document.text}\n"
        "Query:"
    )


# UDAPDR Figure 2, Prompts #1-#5.
STAGE1_PROMPTS = [
    _build_stage1_gbq,
    _build_stage1_vanilla,
    _build_stage1_zero1,
    _build_stage1_zero2,
    _build_stage1_zero3,
]

_BAD_PROMPT_INDEX = 1


def _parse_gbq_good_bad(response_text: str) -> tuple[str, str]:
    """Extract (good, bad) from a GBQ response: first non-bad line is good, first "Bad
    Question:" line is bad; either may be empty.
    """
    good = ""
    bad = ""
    for raw in response_text.strip().split("\n"):
        stripped = raw.strip()
        lowered = stripped.lower()
        if lowered.startswith("bad question"):
            if not bad:
                _, _, rest = stripped.partition(":")
                bad = rest.strip().strip('"\'')
            continue
        if not good:
            cleaned = clean_query_line(raw)
            if cleaned:
                good = cleaned
    return good, bad


def _build_stage2_prompt(
    document: Document,
    demonstrations: list[dict[str, str]],
) -> str:
    """UDAPDR Figure 3: demonstration triples from Stage-1 outputs, then a dangling Good
    Question for the new passage.
    """
    parts: list[str] = []
    for i, demo in enumerate(demonstrations, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Document: {demo['document']}")
        parts.append(f"Good Question: {demo['good_query']}")
        parts.append(f"Bad Question: {demo['bad_query']}")
        parts.append("")
    parts.append(f"Example {len(demonstrations) + 1}:")
    parts.append(f"Document: {document.text}")
    parts.append("Good Question:")
    return "\n".join(parts)


class UdapdrGenerator:
    """UDAPDR two-stage generation: strong model seeds (Stage 1-2), cheap model scales (Stage
    3), one query per passage.
    """

    def __init__(
        self,
        *,
        seed_model: str = settings.SEED_MODEL,
        scale_model: str = settings.GENERATOR_MODEL,
        region: str = settings.REGION,
        temperature: float = settings.GENERATION_TEMPERATURE,
        seed: int = settings.SPLIT_SEED,
        max_workers: int = 5,
    ) -> None:
        self.temperature = temperature
        self.seed = seed
        self.seed_llm = BedrockLLM(model_id=seed_model, region=region, max_workers=max_workers)
        self.scale_llm = BedrockLLM(model_id=scale_model, region=region, max_workers=max_workers)

    def generate(self, documents: list[Document]) -> list[Query]:
        rng = random.Random(self.seed)

        seed_count = min(DEFAULT_SEED_PASSAGES, len(documents))
        seed_indices = sorted(rng.sample(range(len(documents)), seed_count))
        seed_docs = [documents[i] for i in seed_indices]

        logger.info(
            "UDAPDR Stage 1 (seed=%s): %d passages x %d prompts = %d seed calls",
            self.seed_llm.model_id, len(seed_docs), len(STAGE1_PROMPTS),
            len(seed_docs) * len(STAGE1_PROMPTS),
        )

        stage1_doc_idx: list[int] = []
        stage1_prompt_idx: list[int] = []
        stage1_messages: list[list[dict]] = []
        for d_i, doc in enumerate(seed_docs):
            for p_i, prompt_fn in enumerate(STAGE1_PROMPTS):
                stage1_doc_idx.append(d_i)
                stage1_prompt_idx.append(p_i)
                stage1_messages.append([BedrockLLM.user_message(prompt_fn(doc))])

        pbar = tqdm(total=len(stage1_messages), desc="UDAPDR Stage 1: seed generation")
        stage1_responses = self.seed_llm.invoke_batch(
            stage1_messages,
            system=QUERY_GEN_SYSTEM,
            temperature=self.temperature,
            max_tokens=256,
            on_progress=lambda done, total: pbar.update(done - pbar.n),
        )
        pbar.close()

        per_doc: list[dict[str, list[str]]] = [
            {"good": [], "bad": []} for _ in seed_docs
        ]
        for d_i, p_i, resp in zip(stage1_doc_idx, stage1_prompt_idx, stage1_responses):
            if p_i == 0:
                good, bad = _parse_gbq_good_bad(resp.text)
                if good:
                    per_doc[d_i]["good"].append(good)
                if bad:
                    per_doc[d_i]["bad"].append(bad)
            elif p_i == _BAD_PROMPT_INDEX:
                q = first_query(resp.text)
                if q:
                    per_doc[d_i]["bad"].append(q)
            else:
                q = first_query(resp.text)
                if q:
                    per_doc[d_i]["good"].append(q)

        seed_triples: list[dict[str, str]] = []
        for d_i, doc in enumerate(seed_docs):
            goods = per_doc[d_i]["good"]
            bads = per_doc[d_i]["bad"]
            if not goods:
                continue
            good = goods[0]
            if bads:
                bad = bads[0]
            else:
                bad = self._borrow_bad(per_doc, exclude=d_i, rng=rng)
                if not bad:
                    continue
            seed_triples.append(
                {"document": doc.text, "good_query": good, "bad_query": bad}
            )

        logger.info("UDAPDR Stage 1: assembled %d demonstration triples", len(seed_triples))

        if not seed_triples:
            logger.warning("UDAPDR: Stage 1 produced no usable triples; returning []")
            return []

        num_variants = min(DEFAULT_NUM_VARIANTS, len(seed_triples))
        k = min(DEMOS_PER_VARIANT, len(seed_triples))
        demo_sets: list[list[dict[str, str]]] = [
            rng.sample(seed_triples, k) for _ in range(num_variants)
        ]

        logger.info(
            "UDAPDR Stage 2: %d prompt variants, %d demo triples each", num_variants, k,
        )

        stage3_messages: list[list[dict]] = []
        for d_i, doc in enumerate(documents):
            demos = demo_sets[d_i % num_variants]
            stage3_messages.append(
                [BedrockLLM.user_message(_build_stage2_prompt(doc, demos))]
            )

        pbar = tqdm(total=len(stage3_messages), desc="UDAPDR Stage 3: scale generation")
        stage3_responses = self.scale_llm.invoke_batch(
            stage3_messages,
            system=QUERY_GEN_SYSTEM,
            temperature=self.temperature,
            max_tokens=256,
            on_progress=lambda done, total: pbar.update(done - pbar.n),
        )
        pbar.close()

        all_queries: list[Query] = []
        for doc, resp in zip(documents, stage3_responses):
            text = first_query(resp.text)
            if text:
                all_queries.append(make_query(text, doc, "udapdr"))

        logger.info(
            "UDAPDR: %d output queries (%d seed triples, %d variants)",
            len(all_queries), len(seed_triples), num_variants,
        )
        return all_queries

    @staticmethod
    def _borrow_bad(
        per_doc: list[dict[str, list[str]]],
        *,
        exclude: int,
        rng: random.Random,
    ) -> str:
        """A 'bad' query for a passage that produced none: prefers another passage's bad
        candidate, falls back to its good one.
        """
        bad_pool = [
            q for i, d in enumerate(per_doc) if i != exclude for q in d["bad"]
        ]
        if bad_pool:
            return rng.choice(bad_pool)
        good_pool = [
            q for i, d in enumerate(per_doc) if i != exclude for q in d["good"]
        ]
        if good_pool:
            return rng.choice(good_pool)
        return ""
