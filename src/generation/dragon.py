"""DRAGON single-hop structured query generation + rephrasing.

Faithful reimplementation of DRAGON's single-hop pipeline (Shen et al. 2026,
arXiv:2505.10989): extract a clue (supporting sentence) from each document,
generate a seed query from the clue, then apply one equivalence-preserving
reformulation along a logical or completeness axis. The multi-hop
entity-graph branch is dropped: a multi-hop query has no single source
document to serve as its relevance label.
"""

from __future__ import annotations

import logging
import random

from tqdm import tqdm

import settings
from core.llm import BedrockLLM
from core.types import Document, Query
from generation.base import make_query
from generation.parsing import first_query

logger = logging.getLogger(__name__)


def _truncate_doc(text: str, max_tokens: int) -> str:
    """Fallback clue (first `max_tokens` tokens) used only when clue parsing fails; never caps
    the document shown to the model.
    """
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


_CLUE_SYSTEM_PROMPT = (
    "You extract a single supporting sentence (a 'clue') from a document. "
    "A good clue is one self-contained, factual sentence that states a "
    "specific piece of information a reader might want to look up. Output "
    "ONLY that sentence, copied or lightly trimmed from the document, with "
    "no labels, quotation marks, numbering, or explanation."
)


def _build_clue_prompt(document: Document) -> str:
    title = document.title.strip() if document.title else ""
    header = f"Title: {title}\n" if title else ""
    return (
        f"{header}Document: {document.text}\n"
        "Most informative supporting sentence:"
    )


_SEED_SYSTEM_PROMPT = (
    "You are a search query generator. Given a single supporting sentence "
    "(a clue) from a document, write one natural question that this sentence "
    "answers — the kind a real user would type into a search engine. Output "
    "ONLY the question text, with no labels, prefixes, numbering, markdown, "
    "quotation marks, or explanation."
)


def _build_seed_prompt(clue: str) -> str:
    return f"Clue: {clue}\nQuestion this clue answers:"


_LOGICAL_TRANSFORMS: list[tuple[str, str]] = [
    (
        "terminology_substitution",
        "Rewrite the question using domain or technical terminology in place "
        "of plain wording (or vice versa). Do not change what is being asked; "
        "the same clue must still answer it.",
    ),
    (
        "stylistic_variation",
        "Rewrite the question in a different style or register (e.g. more "
        "formal, more colloquial, or more terse). Keep the exact information "
        "need; only the phrasing changes.",
    ),
    (
        "syntactic_reordering",
        "Rewrite the question with its clauses or constituents reordered, or "
        "switching between active and passive voice. The meaning and the "
        "answer stay identical.",
    ),
    (
        "constraint_inclusion",
        "Rewrite the question to make explicit a qualifier that is already "
        "implied by the clue (e.g. naming the entity or context the clue is "
        "about). Do not add any fact not present in the clue; the same clue "
        "must still fully answer the question.",
    ),
    (
        "figurative_phrasing",
        "Rewrite the question with a more idiomatic or figurative turn of "
        "phrase (e.g. a metaphor for the same concept). The literal "
        "information need and its answer are unchanged.",
    ),
]

_COMPLETENESS_TRANSFORMS: list[tuple[str, str]] = [
    (
        "near_synonym_replacement",
        "Rewrite the question by replacing one or two content words with close "
        "synonyms or paraphrases drawn from the clue's topic. Preserve the "
        "information need exactly.",
    ),
    (
        "underspecification",
        "Rewrite the question to reveal LESS of the supporting clue: drop one "
        "specific detail and refer to it more generally, so the query is "
        "shorter and less complete. The same clue must still be a valid "
        "answer; do not introduce any new entity.",
    ),
    (
        "elaboration",
        "Rewrite the question to reveal MORE of the supporting clue: restate "
        "an extra detail that is already present in the clue. Do not add any "
        "fact absent from the clue.",
    ),
    (
        "perspective_shift",
        "Rewrite the question from a different perspective (e.g. first person "
        "to third person, user to provider). Keep the same information need "
        "and the same answer.",
    ),
    (
        "reference_paraphrase",
        "Rewrite the question by paraphrasing how it refers to the entity or "
        "topic of the clue (e.g. a description in place of a name), without "
        "changing which clue answers it.",
    ),
]

_REPHRASE_SYSTEM_PROMPT = (
    "You are a question rewriter. You are given a supporting sentence (a "
    "clue), a question that the clue answers, and a transformation "
    "instruction. Apply the transformation to produce a single rewritten "
    "question that is STILL answered by the same clue. Return only the "
    "rewritten question on one line, with no explanation, quotation marks, "
    "or prefix."
)


def _build_rephrase_prompt(clue: str, seed_query: str, instruction: str) -> str:
    return (
        f"Clue: {clue}\n"
        f"Question: {seed_query}\n"
        f"Transformation: {instruction}\n"
        "Rewritten question (still answered by the clue):"
    )


def _active_transforms(logical: int, completeness: int) -> list[tuple[str, str, str]]:
    """Returns `(axis, name, instruction)` triples for the active transforms, capped per axis
    (full catalogue is 5/5).
    """
    n_log = max(0, int(logical))
    n_comp = max(0, int(completeness))
    triples: list[tuple[str, str, str]] = []
    for name, instr in _LOGICAL_TRANSFORMS[:n_log]:
        triples.append(("logical", name, instr))
    for name, instr in _COMPLETENESS_TRANSFORMS[:n_comp]:
        triples.append(("completeness", name, instr))
    return triples


class DragonGenerator:
    """Single-hop DRAGON: clue extraction -> clue-grounded seed query -> equivalence-preserving
    reformulation.
    """

    def __init__(
        self,
        llm: BedrockLLM,
        *,
        temperature: float = settings.GENERATION_TEMPERATURE,
        seed: int = settings.SPLIT_SEED,
        logical_transforms: int = 5,
        completeness_transforms: int = 5,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.seed = seed
        self.logical_transforms = logical_transforms
        self.completeness_transforms = completeness_transforms

    def generate(self, documents: list[Document]) -> list[Query]:
        if not documents:
            return []

        clue_prompts = [_build_clue_prompt(doc) for doc in documents]
        clue_batches = [[BedrockLLM.user_message(p)] for p in clue_prompts]

        pbar = tqdm(total=len(documents), desc="DRAGON: clue extraction")

        def on_clue_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        clue_responses = self.llm.invoke_batch(
            clue_batches,
            system=_CLUE_SYSTEM_PROMPT,
            temperature=self.temperature,
            on_progress=on_clue_progress,
        )
        pbar.close()

        clued: list[tuple[Document, str]] = []
        for doc, resp in zip(documents, clue_responses):
            clue = first_query(resp.text)
            if not clue:
                clue = _truncate_doc(doc.text, 40)
            if clue:
                clued.append((doc, clue))
        logger.info("DRAGON: extracted %d clues from %d documents", len(clued), len(documents))
        if not clued:
            return []

        seed_prompts = [_build_seed_prompt(clue) for _, clue in clued]
        seed_batches = [[BedrockLLM.user_message(p)] for p in seed_prompts]

        pbar = tqdm(total=len(clued), desc="DRAGON: seed generation")

        def on_seed_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        seed_responses = self.llm.invoke_batch(
            seed_batches,
            system=_SEED_SYSTEM_PROMPT,
            temperature=self.temperature,
            on_progress=on_seed_progress,
        )
        pbar.close()

        seeds: list[tuple[Document, str, str]] = []
        for (doc, clue), resp in zip(clued, seed_responses):
            seed_text = first_query(resp.text)
            if seed_text:
                seeds.append((doc, clue, seed_text))
        logger.info("DRAGON: produced %d seed queries from %d clues", len(seeds), len(clued))
        if not seeds:
            return []

        transforms = _active_transforms(self.logical_transforms, self.completeness_transforms)
        if not transforms:
            logger.info("DRAGON: no active transforms; returning %d seeds", len(seeds))
            return [
                self._make(doc, clue, seed, axis="none", name="none", final=seed)
                for (doc, clue, seed) in seeds
            ]

        rng = random.Random(self.seed)
        plan: list[tuple[str, str, str]] = [rng.choice(transforms) for _ in seeds]

        rephrase_prompts = [
            _build_rephrase_prompt(clue, seed, instr)
            for (_, clue, seed), (_, _, instr) in zip(seeds, plan)
        ]
        rephrase_batches = [[BedrockLLM.user_message(p)] for p in rephrase_prompts]

        pbar = tqdm(total=len(rephrase_batches), desc="DRAGON: rephrasing")

        def on_rephrase_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        rephrase_responses = self.llm.invoke_batch(
            rephrase_batches,
            system=_REPHRASE_SYSTEM_PROMPT,
            temperature=self.temperature,
            on_progress=on_rephrase_progress,
        )
        pbar.close()

        all_queries: list[Query] = []
        axis_counts: dict[str, int] = {}
        for (doc, clue, seed), (axis, name, _instr), resp in zip(seeds, plan, rephrase_responses):
            rewritten = first_query(resp.text)
            if rewritten:
                final, out_axis, out_name = rewritten, axis, name
            else:
                final, out_axis, out_name = seed, "none", "none"
            axis_counts[out_axis] = axis_counts.get(out_axis, 0) + 1
            all_queries.append(
                self._make(doc, clue, seed, axis=out_axis, name=out_name, final=final)
            )

        logger.info("DRAGON: generated %d queries", len(all_queries))
        logger.info("DRAGON: axis distribution: %s", sorted(axis_counts.items()))
        return all_queries

    @staticmethod
    def _make(
        doc: Document,
        clue: str,
        seed: str,
        *,
        axis: str,
        name: str,
        final: str,
    ) -> Query:
        query = make_query(final, doc, "dragon")
        query.metadata["dragon_transform"] = name
        query.metadata["dragon_axis"] = axis
        query.metadata["dragon_clue"] = clue
        query.metadata["dragon_seed_text"] = seed
        return query
