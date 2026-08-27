"""Central configuration: corpora, the frozen retrieval pipeline set, models, seeds, paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RAG_EVAL_RESULTS_DIR", "results"))
CACHE_DIR = Path(os.environ.get("RAG_EVAL_CACHE_DIR", "cache"))

REGION = os.environ.get("RAG_EVAL_BEDROCK_REGION", "us-east-1")

GENERATOR_MODEL = "google.gemma-3-27b-it"
SEED_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
RELEVANCE_JUDGE_MODEL = "qwen.qwen3-32b-v1:0"
OPTIMISER_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
ANSWER_JUDGE_MODEL = "deepseek.v3.2"
REFERENCE_SYNTHESISER_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GENERATION_TEMPERATURE = 0.7

SEEDS = [42, 1337, 2024]
SPLIT_SEED = 42
TRAIN_RATIO = 0.1


@dataclass(frozen=True)
class Corpus:
    name: str
    loader: str
    label: str
    device: str
    max_queries: int | None = None


CORPORA: dict[str, Corpus] = {
    "techqa": Corpus("techqa", "techqa", "TechQA", device="cpu"),
    "nfcorpus": Corpus("nfcorpus", "beir", "NFCorpus", device="mps"),
    "clapnq": Corpus("clapnq", "clapnq", "ClapNQ", device="cpu", max_queries=1000),
    "fiqa": Corpus("fiqa", "beir", "FiQA", device="cpu"),
    "wixqa": Corpus("wixqa", "wixqa", "WixQA", device="mps"),
}

ALL_CORPORA = ["techqa", "nfcorpus", "clapnq", "fiqa", "wixqa"]

ROSTER = [
    "bm25",
    "bm25__ce",
    "colbertv2",
    "d_gte_base",
    "rm3_bm25",
    "rrf_bm25_splade",
    "rrf_splade_d_e5_base",
    "splade",
]

PARADIGMS = {
    "bm25": "lexical",
    "rm3_bm25": "lexical",
    "splade": "learned_sparse",
    "d_gte_base": "dense",
    "rrf_bm25_splade": "fusion",
    "rrf_splade_d_e5_base": "fusion",
    "bm25__ce": "rerank",
    "colbertv2": "late_interaction",
}

TRAIN_PIPELINES = ["rrf_splade_d_e5_base", "splade", "colbertv2", "bm25", "rm3_bm25"]
TEST_PIPELINES = ["d_gte_base", "rrf_bm25_splade", "bm25__ce"]

PRIMARY_METRIC = "ndcg_cut_10"
SECONDARY_METRICS = ["ndcg_cut_5", "map"]
CEILING_METRICS = ["ndcg_cut_10", "ndcg_cut_5", "map", "recip_rank"]

POOL_DEPTH = 10
BOOTSTRAP_DRAWS = 10_000
CEILING_SEED = 12345

GENERATORS = ["naive", "inpars", "promptagator", "duqgen", "udapdr", "dragon"]


@dataclass
class DatasetSpec:
    """Resolved load parameters for one corpus."""

    name: str
    max_queries: int | None = None
    max_corpus: int | None = None
    train_ratio: float = TRAIN_RATIO
    split_seed: int = SPLIT_SEED
    metadata: dict = field(default_factory=dict)


def spec(corpus: str, **overrides) -> DatasetSpec:
    c = CORPORA[corpus]
    params = {"name": c.name, "max_queries": c.max_queries}
    params.update(overrides)
    return DatasetSpec(**params)


def device_for(corpus: str) -> str:
    return CORPORA[corpus].device


def results_path(*parts) -> Path:
    p = RESULTS_DIR.joinpath(*[str(x) for x in parts])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
