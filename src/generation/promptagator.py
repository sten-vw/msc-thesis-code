"""In-domain few-shot generation (Promptagator).

Interleaves up to 8 in-domain (document, query) exemplars, each wrapped in
a task-specific Table 4 template, then asks the LLM to complete the query
for a new document. No official code exists; the paper's FLAN completion
layout ("X"-delimited, base/FLAN continuation) is adapted into an explicit
instruction for an instruction-tuned backbone, keeping the Table-4
wrappers and few-shot exemplars unchanged.

Source: Promptagator (Dai et al., ICLR 2023)
  Paper: arXiv:2209.11755, §3.1 and Appendix Table 4 and Figure 5
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
from generation.prompts import QUERY_GEN_SYSTEM

logger = logging.getLogger(__name__)

# Promptagator Appendix Fig. 5 FLAN example separator.
_SEP = "X"

# Promptagator §3.1: "at most 8 examples".
_MAX_EXAMPLES = 8

# Promptagator Table 4 (Dai et al., ICLR 2023): (e_doc, e_query) wrappers.
TASK_WRAPPERS: dict[str, tuple[str, str]] = {
    "arguana": ("Argument: {text}", "Counter argument: {text}"),
    "fiqa": ("{text}", "{text}"),
    "hotpotqa": ("Evidence: {text}", "Vexed question: {text}"),
    "dbpedia-entity": ("entity: {text}", "query: {text}"),
    "nfcorpus": ("Article: {text}", "Query: {text}"),
    "touche-2020": ("{text}", "Debate: {text}"),
    "trec-covid": ("{text}", "Question: {text}"),
    "scifact": ("{text}", "Finding: {text}"),
    "scidocs": ("{text}", "The passage is about {text}"),
    "fever": ("{text}", "Is it true that {text}"),
    "climate-fever": ("{text}", "Is it true that {text}"),
}

_DEFAULT_WRAPPER = ("{text}", "Question: {text}")


def _get_wrappers(dataset_name: str) -> tuple[str, str]:
    """Table 4 wrappers for a dataset name; unknown datasets fall back to a generic QA
    template.
    """
    return TASK_WRAPPERS.get(dataset_name.lower(), _DEFAULT_WRAPPER)


def _wrap(template: str, text: str) -> str:
    return template.replace("{text}", text.strip())


def _query_cue(e_query: str) -> str:
    """The query-wrapper prefix before ``{text}``, or a generic ``Query:`` cue for a bare
    wrapper.
    """
    prefix = e_query.split("{text}")[0].strip()
    return prefix or "Query:"


def _doc_block(e_doc: str, text: str) -> str:
    """Render a document under its Table 4 wrapper; a bare ``{text}`` wrapper gets a generic
    ``Document:`` label.
    """
    seg = _wrap(e_doc, text)
    return seg if e_doc.strip() != "{text}" else f"Document: {seg}"


def _query_block(e_query: str, text: str) -> str:
    """Render an example query under its Table 4 wrapper; a bare ``{text}`` wrapper gets the
    generic ``Query:`` cue.
    """
    seg = _wrap(e_query, text)
    return seg if e_query.strip() != "{text}" else f"{_query_cue(e_query)} {seg}"


def _build_prompt(
    document: Document,
    examples: list[tuple[str, str]],
    e_doc: str,
    e_query: str,
) -> str:
    """Labelled (document, query) demonstrations, an instruction, then the target document
    ending on the query cue.
    """
    blocks = [
        f"{_doc_block(e_doc, doc_text)}\n{_query_block(e_query, query_text)}"
        for doc_text, query_text in examples
    ]
    demo = "\n\n".join(blocks)
    target = _doc_block(e_doc, document.text)
    cue = _query_cue(e_query)
    return (
        "Below are example documents from this collection, each followed by a "
        "query that it answers:\n\n"
        f"{demo}\n\n"
        "Now write one query, in the same style as the examples, that the "
        "following document answers. Output only the query.\n\n"
        f"{target}\n{cue}"
    )


class PromptagatorGenerator:
    """Promptagator few-shot query generation (Dai et al., ICLR 2023)."""

    def __init__(
        self,
        llm: BedrockLLM,
        *,
        dataset_name: str = "",
        temperature: float = settings.GENERATION_TEMPERATURE,
        seed: int = settings.SPLIT_SEED,
        few_shot_count: int = 8,
    ) -> None:
        self.llm = llm
        self.dataset_name = dataset_name
        self.temperature = temperature
        self.seed = seed
        self.few_shot_count = few_shot_count

    def generate(
        self,
        documents: list[Document],
        *,
        few_shot_queries: list[Query],
        few_shot_docs: dict[str, Document],
        num_queries_per_doc: int = 1,
    ) -> list[Query]:
        if not few_shot_queries:
            raise ValueError(
                "Promptagator requires real (document, query) example pairs. "
                "Pass train-split queries via few_shot_queries."
            )
        if not few_shot_docs:
            raise ValueError(
                "Promptagator requires the corpus to look up the document text "
                "of each few-shot example. Pass few_shot_docs as a "
                "{doc_id: Document} dict."
            )

        usable = [
            q
            for q in few_shot_queries
            if q.source_doc_id and q.source_doc_id in few_shot_docs
        ]
        dropped = len(few_shot_queries) - len(usable)
        if dropped:
            logger.warning(
                "Promptagator: %d/%d few-shot queries have no resolvable source "
                "document and were skipped as exemplars.",
                dropped,
                len(few_shot_queries),
            )
        if not usable:
            raise ValueError(
                "Promptagator: none of the few-shot queries resolve to a "
                "document in few_shot_docs; cannot build (document, query) "
                "example pairs."
            )

        e_doc, e_query = _get_wrappers(self.dataset_name)
        rng = random.Random(self.seed)
        k = min(self.few_shot_count, _MAX_EXAMPLES, len(usable))

        call_docs: list[Document] = []
        message_batches: list[list[dict]] = []
        for doc in documents:
            chosen = rng.sample(usable, k)
            examples = [
                (few_shot_docs[q.source_doc_id].text, q.text) for q in chosen
            ]
            prompt = _build_prompt(doc, examples, e_doc, e_query)
            for _ in range(num_queries_per_doc):
                message_batches.append([BedrockLLM.user_message(prompt)])
                call_docs.append(doc)

        total_calls = len(message_batches)
        pbar = tqdm(total=total_calls, desc="Promptagator few-shot")

        def on_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        responses = self.llm.invoke_batch(
            message_batches,
            system=QUERY_GEN_SYSTEM,
            temperature=self.temperature,
            max_tokens=256,
            on_progress=on_progress,
        )
        pbar.close()

        cue = _query_cue(e_query)
        query_label = cue if cue.endswith(":") else ""
        all_queries: list[Query] = []
        n_failed = 0
        for doc, response in zip(call_docs, responses):
            text = self._extract_query(response.text, query_label)
            if not text:
                n_failed += 1
                continue
            all_queries.append(make_query(text, doc, "promptagator"))

        logger.info(
            "Promptagator: %d queries from %d documents "
            "(k=%d examples, %d/%d generation failures, dataset=%r)",
            len(all_queries),
            len(documents),
            k,
            n_failed,
            total_calls,
            self.dataset_name or "<unset -> generic wrapper>",
        )
        return all_queries

    @staticmethod
    def _extract_query(raw: str, query_label: str) -> str:
        """Recover q_hat: strip a wrapper-label echo and a trailing FLAN "X" separator, reduce
        to one line.
        """
        raw = raw.strip()
        if query_label and raw.lower().startswith(query_label.lower()):
            raw = raw[len(query_label):].lstrip(" :").strip()
        if f" {_SEP} " in raw:
            raw = raw.split(f" {_SEP} ", 1)[0].strip()
        if raw.endswith(f" {_SEP}"):
            raw = raw[: -len(_SEP)].strip()
        text = first_query(raw)
        if text and query_label and text.lower().startswith(query_label.lower()):
            text = text[len(query_label):].lstrip(" :").strip()
        return text
