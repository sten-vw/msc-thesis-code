"""Corpus loaders for the five public evaluation corpora and the production system.

TechQA (Castelli et al. 2020), NFCorpus (Boteva et al. 2016), ClapNQ (Rosenthal
et al. 2025), FiQA (Maia et al. 2018) and WixQA (Cohen et al. 2025). NFCorpus and
FiQA are read in BEIR format (Thakur et al. 2021).
"""

from __future__ import annotations

import ast
import logging
import random
import re

from core.types import Document, Query, RelevanceJudgments
from settings import CORPORA, DatasetSpec

logger = logging.getLogger(__name__)

_BEIR = {
    "nfcorpus": {"hf_id": "BeIR/nfcorpus", "qrels_hf_id": "BeIR/nfcorpus-qrels"},
    "fiqa": {"hf_id": "BeIR/fiqa", "qrels_hf_id": "BeIR/fiqa-qrels"},
}

_CLAPNQ_CORPUS_HF_ID = "PrimeQA/clapnq_passages"
_CLAPNQ_QA_HF_ID = "PrimeQA/clapnq"
_TECHQA_HF_ID = "nvidia/TechQA-RAG-Eval"
_WIXQA_HF_ID = "Wix/WixQA"
_WIXQA_SPLIT = "expertwritten"


def load_corpus(
    spec: DatasetSpec,
) -> tuple[dict[str, Document], dict[str, Query], RelevanceJudgments]:
    """Return (corpus, queries, qrels) for the named corpus."""
    kind = CORPORA[spec.name].loader
    if kind == "beir":
        return _load_beir(spec)
    if kind == "clapnq":
        return _load_clapnq(spec)
    if kind == "techqa":
        return _load_techqa(spec)
    if kind == "wixqa":
        return _load_wixqa(spec)
    raise ValueError(f"Unknown corpus {spec.name!r}. Available: {sorted(CORPORA)}")


def max_relevance_grade(qrels: RelevanceJudgments) -> int:
    return max((s for docs in qrels.judgments.values() for s in docs.values()), default=1)


def _build_corpus(corpus_ds, keep_ids: set[str], max_corpus: int | None, seed: int):
    if max_corpus is None:
        return {
            str(r["_id"]): Document(
                doc_id=str(r["_id"]),
                title=r.get("title", "") or "",
                text=r.get("text", "") or "",
            )
            for r in corpus_ds
        }

    rng = random.Random(seed)
    budget = max(max_corpus - len(keep_ids), 0)
    corpus: dict[str, Document] = {}
    filler: list[Document] = []
    seen_nongold = 0
    for r in corpus_ds:
        doc_id = str(r["_id"])
        doc = Document(
            doc_id=doc_id, title=r.get("title", "") or "", text=r.get("text", "") or "",
        )
        if doc_id in keep_ids:
            corpus[doc_id] = doc
            continue
        seen_nongold += 1
        if len(filler) < budget:
            filler.append(doc)
        elif budget > 0:
            j = rng.randint(0, seen_nongold - 1)
            if j < budget:
                filler[j] = doc
    for doc in filler:
        corpus[doc.doc_id] = doc
    return corpus


def _assemble_beir_format(corpus_ds, queries_ds, qrels_rows, spec: DatasetSpec):
    judgments: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        judgments.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    queries: dict[str, Query] = {}
    for row in queries_ds:
        qid = str(row["_id"])
        if qid not in judgments:
            continue
        best_doc_id = max(judgments[qid], key=judgments[qid].get)
        queries[qid] = Query(query_id=qid, text=row["text"], source_doc_id=best_doc_id)

    if spec.max_queries and len(queries) > spec.max_queries:
        qids = list(queries.keys())[: spec.max_queries]
        queries = {qid: queries[qid] for qid in qids}
        judgments = {qid: judgments[qid] for qid in queries}

    gold_ids = {d for docs in judgments.values() for d in docs}
    corpus = _build_corpus(corpus_ds, gold_ids, spec.max_corpus, spec.split_seed)
    logger.info("%s: %d docs, %d judged queries", spec.name, len(corpus), len(queries))
    return corpus, queries, RelevanceJudgments(judgments=judgments)


def _load_beir(spec: DatasetSpec):
    import datasets as hf_datasets

    info = _BEIR[spec.name]
    corpus_ds = hf_datasets.load_dataset(info["hf_id"], "corpus", split="corpus")
    queries_ds = hf_datasets.load_dataset(info["hf_id"], "queries", split="queries")
    qrels_ds = hf_datasets.load_dataset(info["qrels_hf_id"], split="test")
    return _assemble_beir_format(corpus_ds, queries_ds, qrels_ds, spec)


def _parse_id_list(field) -> list[str]:
    if field is None:
        return []
    if isinstance(field, (list, tuple)):
        return [str(x) for x in field]
    s = str(field).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
    except (ValueError, SyntaxError):
        pass
    return [tok for tok in re.split(r"[,\s]+", s.strip("[]")) if tok]


def _load_wixqa(spec: DatasetSpec):
    import datasets as hf_datasets

    corpus_ds = hf_datasets.load_dataset(_WIXQA_HF_ID, "wix_kb_corpus", split="train")
    corpus = {
        str(row["id"]): Document(
            doc_id=str(row["id"]),
            title=row.get("title", "") or "",
            text=row.get("contents", "") or "",
        )
        for row in corpus_ds
    }

    qa_ds = hf_datasets.load_dataset(_WIXQA_HF_ID, f"wixqa_{_WIXQA_SPLIT}", split="train")
    queries: dict[str, Query] = {}
    judgments: dict[str, dict[str, int]] = {}
    for i, row in enumerate(qa_ds):
        qid = str(row.get("id", i))
        gold = [a for a in _parse_id_list(row.get("article_ids")) if a in corpus]
        if not gold:
            continue
        judgments[qid] = {a: 1 for a in gold}
        metadata = {"reference_answer": row["answer"]} if row.get("answer") else {}
        queries[qid] = Query(
            query_id=qid, text=row["question"], source_doc_id=gold[0], metadata=metadata,
        )

    if spec.max_queries and len(queries) > spec.max_queries:
        qids = list(queries.keys())[: spec.max_queries]
        queries = {qid: queries[qid] for qid in qids}
        judgments = {qid: judgments[qid] for qid in queries}

    logger.info("wixqa: %d docs, %d judged queries", len(corpus), len(queries))
    return corpus, queries, RelevanceJudgments(judgments=judgments)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _load_clapnq(spec: DatasetSpec):
    import datasets as hf_datasets

    corpus_ds = hf_datasets.load_dataset(_CLAPNQ_CORPUS_HF_ID, split="train")
    corpus: dict[str, Document] = {}
    text_to_pid: dict[tuple[str, str], str] = {}
    for row in corpus_ds:
        pid = str(row["id"])
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        corpus[pid] = Document(doc_id=pid, title=title, text=text)
        text_to_pid[(_normalize(title), _normalize(text))] = pid

    if spec.max_corpus and len(corpus) > spec.max_corpus:
        doc_ids = list(corpus.keys())[: spec.max_corpus]
        corpus = {did: corpus[did] for did in doc_ids}

    queries: dict[str, Query] = {}
    judgments: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        qa_ds = hf_datasets.load_dataset(_CLAPNQ_QA_HF_ID, split=split)
        for row in qa_ds:
            qid = str(row["id"])
            metadata: dict = {}
            outputs = row.get("output") or []
            if outputs:
                if outputs[0].get("answer"):
                    metadata["reference_answer"] = outputs[0]["answer"]
                if outputs[0].get("selected_sentences"):
                    metadata["selected_sentences"] = outputs[0]["selected_sentences"]

            source_doc_id = None
            passages = row.get("passages") or []
            if passages:
                gold = passages[0]
                pid = text_to_pid.get(
                    (_normalize(gold.get("title", "") or ""),
                     _normalize(gold.get("text", "") or ""))
                )
                if pid is not None:
                    source_doc_id = pid
                    judgments[qid] = {pid: 1}
            queries[qid] = Query(
                query_id=qid, text=row["input"],
                source_doc_id=source_doc_id, metadata=metadata,
            )

    queries = {qid: q for qid, q in queries.items() if qid in judgments}
    if spec.max_queries and len(queries) > spec.max_queries:
        qids = list(queries.keys())[: spec.max_queries]
        queries = {qid: queries[qid] for qid in qids}
        judgments = {qid: judgments[qid] for qid in queries}

    logger.info("clapnq: %d docs, %d judged queries", len(corpus), len(queries))
    return corpus, queries, RelevanceJudgments(judgments=judgments)


def _load_techqa(spec: DatasetSpec):
    import zipfile

    import datasets as hf_datasets
    from huggingface_hub import hf_hub_download

    corpus: dict[str, Document] = {}
    zip_path = hf_hub_download(_TECHQA_HF_ID, "corpus.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            doc_id = name.split("/")[-1]
            raw = zf.read(name).decode("utf-8", "replace")
            lines = raw.splitlines()
            if lines and lines[0].startswith("Title:"):
                title = lines[0][len("Title:"):].strip()
                body = "\n".join(lines[1:]).strip()
            else:
                title, body = "", raw.strip()
            corpus[doc_id] = Document(doc_id=doc_id, title=title, text=body)

    ds = hf_datasets.load_dataset(_TECHQA_HF_ID, split="train")
    queries: dict[str, Query] = {}
    judgments: dict[str, dict[str, int]] = {}
    for row in ds:
        if bool(row.get("is_impossible", False)):
            continue
        qrel = {
            (ctx.get("filename") or "").strip(): 1
            for ctx in (row.get("contexts") or [])
            if (ctx.get("filename") or "").strip() in corpus
        }
        if not qrel:
            continue
        qid = str(row["id"])
        metadata: dict = {}
        if (row.get("answer") or "").strip():
            metadata["reference_answer"] = row["answer"].strip()
        queries[qid] = Query(
            query_id=qid, text=(row.get("question") or "").strip(),
            source_doc_id=next(iter(qrel)), metadata=metadata,
        )
        judgments[qid] = qrel

    if spec.max_corpus and len(corpus) > spec.max_corpus:
        doc_ids = list(corpus.keys())[: spec.max_corpus]
        corpus = {did: corpus[did] for did in doc_ids}
        for qid in list(judgments):
            judgments[qid] = {d: s for d, s in judgments[qid].items() if d in corpus}
            if not judgments[qid]:
                judgments.pop(qid)
                queries.pop(qid, None)

    if spec.max_queries and len(queries) > spec.max_queries:
        rng = random.Random(spec.split_seed)
        picked = rng.sample(sorted(queries), spec.max_queries)
        queries = {qid: queries[qid] for qid in picked}
        judgments = {qid: judgments[qid] for qid in picked}

    logger.info("techqa: %d docs, %d judged queries", len(corpus), len(queries))
    return corpus, queries, RelevanceJudgments(judgments=judgments)
