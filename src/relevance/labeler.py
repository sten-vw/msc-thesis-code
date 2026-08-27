"""LLM-based relevance labeling for synthetic query-document pairs.

Follows the Thomas et al. (2023) relevance-rating prompt as applied to
synthetic test collections by Rahmani et al. (2024). Two qrel modes: sparse
(`create_sparse_qrels`, label-free, query relevant only to its source doc)
and labeled (`create_labeled_qrels`, LLM grades every query-document pair in
a shared, cross-pipeline pool on a graded scale).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from tqdm import tqdm

from core.llm import BedrockLLM
from core.types import Document, Query, RelevanceJudgments

logger = logging.getLogger(__name__)

# Thomas et al. (2023) / Rahmani et al. (2024) relevance-rating prompt.
# https://github.com/rahmanidashti/SyntheticTestCollections
_THOMAS_PROMPT_0_3 = """\
You are a search quality rater evaluating the relevance of passages. Given a \
query and a passage, you must provide a score on an integer scale of 0 to 3 \
with the following meanings:

3 = Perfectly relevant: The passage is dedicated to the query and contains \
the exact answer.
2 = Highly relevant: The passage has some answer for the query, but the \
answer may be a bit unclear, or hidden amongst extraneous information.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query.

Assume that you are writing an answer to the query. If the passage seems to \
be related to the query but does not include any answer to the query, mark \
it 1. If you would use any of the information contained in the passage in \
such an answer, mark it 2. If the passage is primarily about the query, or \
contains vital information about the topic, mark it 3. Otherwise, mark it 0.

A person has typed [{query}] into a search engine.

Result
Consider the following passage.
—BEGIN Passage CONTENT—
{passage}
—END Passage CONTENT—

Instructions
Consider the underlying intent of the search, and decide on a final score \
of the relevancy of query to the passage given the context.
Score:"""

_THOMAS_PROMPT_0_1 = """\
You are a search quality rater evaluating the relevance of passages. Given a \
query and a passage, you must provide a binary relevance score:

1 = Relevant: The passage contains an answer to the query, or contains vital \
information about the topic.
0 = Not relevant: The passage does not answer the query and is not about the \
topic.

A person has typed [{query}] into a search engine.

Result
Consider the following passage.
—BEGIN Passage CONTENT—
{passage}
—END Passage CONTENT—

Instructions
Consider the underlying intent of the search, and decide on a final score \
of the relevancy of query to the passage given the context. Respond with \
only the digit 0 or 1 and nothing else.
Score:"""

_THOMAS_PROMPT_0_2 = """\
You are a search quality rater evaluating the relevance of passages. Given a \
query and a passage, you must provide a score on an integer scale of 0 to 2 \
with the following meanings:

2 = Very relevant: The passage has an answer for the query, or contains \
vital information about the topic.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query.

Assume that you are writing an answer to the query. If the passage seems to \
be related to the query but does not include any answer to the query, mark \
it 1. If you would use any of the information contained in the passage in \
such an answer, or the passage is primarily about the query, mark it 2. \
Otherwise, mark it 0.

A person has typed [{query}] into a search engine.

Result
Consider the following passage.
—BEGIN Passage CONTENT—
{passage}
—END Passage CONTENT—

Instructions
Consider the underlying intent of the search, and decide on a final score \
of the relevancy of query to the passage given the context.
Score:"""

_PROMPTS = {1: _THOMAS_PROMPT_0_1, 2: _THOMAS_PROMPT_0_2, 3: _THOMAS_PROMPT_0_3}


def _build_relevance_prompt(query: str, passage: str, max_score: int) -> str:
    """Build the relevance labeling prompt for the given scale."""
    template = _PROMPTS.get(max_score)
    if template is None:
        logger.warning("No prompt template for max_score=%d, falling back to 0-3", max_score)
        template = _THOMAS_PROMPT_0_3
    return template.format(query=query, passage=passage)


def create_sparse_qrels(queries: list[Query]) -> RelevanceJudgments:
    """Binary relevance from source_doc_id: each query is relevant only to the document it was
    generated from.
    """
    judgments: dict[str, dict[str, int]] = {}
    for q in queries:
        if q.source_doc_id:
            judgments[q.query_id] = {q.source_doc_id: 1}
    return RelevanceJudgments(judgments=judgments)


def _load_label_cache(cache_path: Path) -> dict[tuple[str, str], int]:
    """Load a (query_id, doc_id) -> score jsonl cache. Missing file -> empty."""
    cache: dict[tuple[str, str], int] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cache[(rec["query_id"], rec["doc_id"])] = int(rec["score"])
    return cache


def create_labeled_qrels(
    queries: list[Query],
    pool: dict[str, list[str]],
    corpus: dict[str, Document],
    llm: BedrockLLM,
    *,
    max_score: int = 2,
    cache_path: str | Path | None = None,
) -> RelevanceJudgments:
    """Grade every (query, document) pair in a shared, cross-pipeline pool (built by
    `retrieval.roster.pool_from_runs`) on a 0-`max_score` scale, where `max_score` matches
    the corpus's native qrel scale; `cache_path`, if given, is a resume-safe JSONL cache
    keyed by (query_id, doc_id).
    """
    query_map = {q.query_id: q for q in queries}
    cache_path = Path(cache_path) if cache_path else None
    cached = _load_label_cache(cache_path) if cache_path else {}

    pairs: list[tuple[str, str]] = []
    judgments: dict[str, dict[str, int]] = {}
    for qid, doc_ids in pool.items():
        for did in doc_ids:
            if (qid, did) in cached:
                judgments.setdefault(qid, {})[did] = cached[(qid, did)]
            else:
                pairs.append((qid, did))

    if not pairs:
        logger.info("All %d pooled pairs served from label cache %s",
                    sum(len(v) for v in judgments.values()), cache_path)
        return RelevanceJudgments(judgments=judgments)

    logger.info(
        "Generating relevance labels for %d query-document pairs "
        "(%d cached, %d queries, scale 0-%d)",
        len(pairs), len(cached), len(pool), max_score,
    )

    jobs: list[tuple[str, str, str]] = []
    for qid, did in pairs:
        query = query_map.get(qid)
        doc = corpus.get(did)
        query_text = query.text if query else ""
        passage_text = f"{doc.title}\n{doc.text}" if doc else ""
        if len(passage_text) > 16000:
            passage_text = passage_text[:16000] + "..."
        prompt = _build_relevance_prompt(query_text, passage_text, max_score)
        jobs.append((qid, did, prompt))

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_fh = None
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_fh = cache_path.open("a")
    write_lock = threading.Lock()
    errors: list[tuple[str, str, str]] = []

    def _label_one(job: tuple[str, str, str]) -> tuple[str, str, int]:
        qid, did, prompt = job
        resp = llm.invoke([BedrockLLM.user_message(prompt)], temperature=0.0, max_tokens=16)
        return qid, did, _parse_relevance_score(resp.text, max_score)

    pbar = tqdm(total=len(jobs), desc="Relevance labeling")
    with ThreadPoolExecutor(max_workers=max(llm._max_workers, 1)) as executor:
        futures = {executor.submit(_label_one, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            pbar.update(1)
            try:
                qid, did, score = fut.result()
            except Exception as e:
                errors.append((job[0], job[1], repr(e)))
                continue
            judgments.setdefault(qid, {})[did] = score
            if cache_fh:
                with write_lock:
                    cache_fh.write(
                        json.dumps({"query_id": qid, "doc_id": did, "score": score}) + "\n")
                    cache_fh.flush()
    pbar.close()
    if cache_fh:
        cache_fh.close()
    if errors:
        logger.warning(
            "%d/%d label calls failed after retries (%d cached this pass); "
            "left uncached, rerun to resume only the failures. e.g. %s",
            len(errors), len(jobs), len(jobs) - len(errors), errors[:3])

    n_labeled = sum(len(docs) for docs in judgments.values())
    n_nonzero = sum(
        1 for docs in judgments.values() for s in docs.values() if s > 0
    )
    logger.info(
        "Relevance labeling complete: %d labels, %d non-zero (%.1f%%)",
        n_labeled,
        n_nonzero,
        100.0 * n_nonzero / n_labeled if n_labeled else 0,
    )

    return RelevanceJudgments(judgments=judgments)


def _parse_relevance_score(text: str, max_score: int = 2) -> int:
    """Parse a relevance score from LLM output, clamped to [0, max_score]."""
    text = text.strip()
    match = re.search(r"\b(\d)\b", text)
    if match:
        return min(int(match.group(1)), max_score)
    logger.warning("Failed to parse relevance score from: %r, defaulting to 0", text)
    return 0
