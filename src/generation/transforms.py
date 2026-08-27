"""Query transforms: paraphrase and de-lexicalisation of an existing synthetic pool.

Rewrites each query's surface form while keeping its information need,
via an LLM with per-query disk caching. Holds the prompt templates for
the naive-paraphrase control, the no-source-chunk-access arms, and the
de-lexicalisation rungs, plus the response parser and cached paraphrase
engine. Identifiers (names, numbers, codes, dates) are always kept.
"""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from metrics.lexical import _STOP, content_set

POOL_COLS = ["query_id", "text", "source_doc_id", "answerable", "orig_text", "sem_cos", "rep"]

ID_RULE = ("Keep any names, numbers, codes, dates, and other specific identifiers "
           "exactly as written (do not change, drop, or invent any). ")

P0_INSTR = "Rephrase the question so it asks for exactly the same information."
P0_PROMPT = (P0_INSTR + " Keep any names, numbers, codes, dates, and other specific "
             "identifiers exactly as written (do not change, drop, or invent any). "
             "Output only the single rephrased question.\n\nQuestion: {q}")
P0_HASH = hashlib.md5(P0_INSTR.encode()).hexdigest()[:10]

EX8_PROMPT = (
    "Here are examples of questions real users typed into a search box:\n{ex}\n\n"
    "Rewrite the question below so it reads like one of these users wrote it — same "
    "length, tone, and wording habits as the examples. Ask for EXACTLY the same "
    "information. " + ID_RULE + "Add no new specifics. Output only the question."
    "\n\nQuestion: {q}"
)
REG_PROMPT = (
    "Rewrite the text below as a short question a real, non-expert user would type. "
    "Use everyday language for the ordinary words. " + ID_RULE +
    "Keep the same information need; do not invent new details. Output only the question."
    "\n\nQuestion: {q}"
)
NEED_PROMPT_1 = (
    "In one sentence, describe the underlying problem or information need behind this "
    "question, in your own words (do not reuse the question's distinctive wording). " +
    ID_RULE + "Output only the sentence.\n\nQuestion: {q}"
)
NEED_PROMPT_2 = (
    "A user has this problem: {need}\n\nWrite the short question they would type into a "
    "search box about it. " + ID_RULE + "Output only the question."
)

BASE_DELEX_PROMPT = (
    "Rephrase the question below so it asks for EXACTLY the same information. "
    "Keep any names, numbers, codes, dates, and other specific identifiers exactly as written "
    "(do not change, drop, or invent any). Reword only the ordinary descriptive words. "
    "In particular do not reuse these words: {ban}. Add no new specifics. "
    "Output only the single rephrased question.\n\nQuestion: {q}"
)
REGISTER_DELEX_PROMPT = (
    "Rewrite the text below as a short question a real, non-expert user would type. Use everyday "
    "language for the ordinary words, but keep any names, numbers, codes, dates, and other "
    "specific identifiers exactly as written (do not drop or invent any). Do not reuse these words: "
    "{ban}. Keep the same information need; do not invent new details. Output only the question."
    "\n\nQuestion: {q}"
)
RUNG_CFG = {
    "R1": {"ban_frac": 0.15, "register": False},
    "R2": {"ban_frac": 0.40, "register": False},
    "R3": {"ban_frac": 0.70, "register": False},
    "R4": {"ban_frac": 1.00, "register": True},
}
MAX_BAN = 25


def extract_question(text: str) -> str:
    """Pull the paraphrased question from a verbose reply: first line ending in '?', else the
    first non-empty line.
    """
    if not text or not text.strip():
        return ""
    lines = [re.sub(r'^[\s>*\-#"\'.0-9)]+', "", ln).strip().strip('"').strip()
             for ln in text.strip().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("**")]
    for ln in lines:
        if ln.endswith("?"):
            return ln
    return lines[0] if lines else ""


def is_identifier(tok: str) -> bool:
    """Codes / acronyms / versions that link a query to its gold doc (never banned)."""
    if any(c.isdigit() for c in tok):
        return True
    return tok.isupper() and len(tok) >= 2


def ban_terms(query_text: str, chunk_cset: set[str], max_ban: int = MAX_BAN) -> list[str]:
    """Descriptive (non-identifier) query words overlapping the source chunk, longest-first."""
    seen, out = set(), []
    for raw in re.findall(r"[A-Za-z0-9.]+", query_text):
        low = raw.lower().strip(".")
        if not low or low in _STOP or len(low) == 1 or low in seen:
            continue
        seen.add(low)
        if is_identifier(raw.strip(".")):
            continue
        if content_set(low) & chunk_cset:
            out.append(low)
    out.sort(key=len, reverse=True)
    return out[:max_ban]


def paraphrase_pool(base_df: pd.DataFrame, prompt_fn, cache_path_fn, *,
                    llm=None, embed=None, temperature: float = 0.0,
                    max_tokens: int = 140, workers: int = 48, rep: int = 0,
                    post=extract_question, no_llm: bool = False) -> pd.DataFrame:
    """Rewrite every query in `base_df` via `prompt_fn`, caching each result at
    `cache_path_fn(query_id)`; returns a DataFrame with `POOL_COLS`.
    """
    rows = list(base_df.itertuples())
    out = [None] * len(rows)
    miss = []
    for i, r in enumerate(rows):
        cp = Path(cache_path_fn(r.query_id))
        if cp.exists():
            out[i] = cp.read_text()
        elif no_llm:
            out[i] = r.text
        else:
            cp.parent.mkdir(parents=True, exist_ok=True)
            miss.append((i, prompt_fn(r.text), cp, r.text))

    if miss and not no_llm:
        if llm is None:
            raise ValueError("paraphrase_pool: llm required when there are cache misses")
        from core.llm import BedrockLLM

        errs = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut = {
                pool.submit(llm.invoke, [BedrockLLM.user_message(prompt)],
                            temperature=temperature, max_tokens=max_tokens): (i, cp, orig)
                for (i, prompt, cp, orig) in miss
            }
            for f in as_completed(fut):
                i, cp, orig = fut[f]
                try:
                    resp = f.result()
                except Exception as e:
                    errs.append((i, repr(e)))
                    continue
                txt = post(resp.text) or orig
                out[i] = txt
                cp.write_text(txt)
        if errs:
            raise RuntimeError(
                f"{len(errs)}/{len(miss)} paraphrase calls failed (e.g. {errs[:3]}); "
                f"{len(miss) - len(errs)} cached — rerun to resume")

    recs = [
        {"query_id": r.query_id, "text": out[i], "source_doc_id": r.source_doc_id,
         "answerable": bool(r.answerable), "orig_text": r.text, "rep": rep}
        for i, r in enumerate(rows)
    ]
    rdf = pd.DataFrame(recs)
    if embed is not None and len(rdf):
        eo = embed.encode(rdf["orig_text"].tolist(), normalize_embeddings=True)
        ep = embed.encode(rdf["text"].tolist(), normalize_embeddings=True)
        rdf["sem_cos"] = (eo * ep).sum(1)
    else:
        rdf["sem_cos"] = np.nan
    return rdf[POOL_COLS]
