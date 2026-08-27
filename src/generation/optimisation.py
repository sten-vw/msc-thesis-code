"""Generate-to-separate: prompt optimisation for discriminative synthetic queries.

A label-free OPRO loop (Yang et al., ICLR 2024) over a generation directive
injected into a doc->query scaffold, steering a generator toward queries that
separate a retrieval roster (`sep`), minimise lexical coverage (`cov`), or
match a real-query coverage anchor (`align`). Fitness is computed only from
synthetic-measurable quantities, gated by a smooth hinge on retrievability
rather than a hard cutoff. The expensive generate+retrieve+score step is
injected by the caller as `evaluate_fn(directive) -> Candidate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

from analysis.matrices import query_discriminativity
from core.llm import BedrockLLM
from generation.inpars import FEW_SHOT_EXAMPLES
from generation.prompts import QUERY_GEN_SYSTEM
from metrics.lexical import content_set, coverage

OBJECTIVES = {
    "sep": "the average spread (standard deviation) of a query's nDCG@10 across the "
           "retrieval pipelines",
    "cov": "the question set's average query-to-source lexical coverage — the fraction of "
           "a question's content words that also appear in its source passage",
    "align": "the question set's average query-to-source lexical coverage — the fraction "
             "of a question's content words that also appear in its source passage",
}

_SENSE = {"sep": "maximize", "cov": "minimize", "align": "match"}


def build_gen_prompt(doc_text: str, directive: str = "", scaffold: str = "g0") -> str:
    """Doc->query prompt on `scaffold` ("g0" naive zero-shot, "g1" InPars few-shot); an empty
    `directive` reproduces the base generator's output.
    """
    if scaffold == "g0":
        parts = [f"Document:\n{doc_text}", ""]
        if directive.strip():
            parts.append(f"Instruction for the question you write: {directive.strip()}")
            parts.append("")
        parts.append("Write a query that this document answers. Output only the query.")
        return "\n".join(parts)
    parts = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Document: {ex['document']}")
        parts.append(f"Good Question: {ex['good_query']}")
        parts.append(f"Bad Question: {ex['bad_query']}")
        parts.append("")
    if directive.strip():
        parts.append(f"Instruction for the question you write: {directive.strip()}")
        parts.append("")
    parts.append(f"Example {len(FEW_SHOT_EXAMPLES) + 1}:")
    parts.append(f"Document: {doc_text}")
    parts.append("Good Question:")
    return "\n".join(parts)


def gen_system(scaffold: str = "g0") -> str | None:
    """System prompt for the scaffold: None for "g0" (naive baseline), the shared persona
    otherwise.
    """
    return None if scaffold == "g0" else QUERY_GEN_SYSTEM


GEN_SYSTEM = QUERY_GEN_SYSTEM


def pool_coverages(texts: list[str], source_texts: list[str]) -> np.ndarray:
    """Per-query query->source content-word coverage (`metrics.lexical.coverage`)."""
    out = []
    for q, d in zip(texts, source_texts):
        out.append(coverage(content_set(q), content_set(d)))
    return np.array(out, dtype=float)


def real_coverage_anchor(query_texts: list[str], source_texts: list[str]) -> float:
    """Mean query->source coverage over real user questions; the target the `align` objective
    steers toward.
    """
    return float(np.nanmean(pool_coverages(query_texts, source_texts)))


def pool_objectives(M_train: np.ndarray, covs: np.ndarray) -> dict:
    """Label-free objectives from a train-subset sparse-nDCG matrix: `sep` mean per-query
    spread, `cov` mean coverage (lower better), `mean_ndcg` retrievability guard.
    """
    crit = query_discriminativity(M_train)
    return {
        "sep": float(np.mean(crit["std"])),
        "cov": float(np.nanmean(covs)),
        "mean_ndcg": float(np.mean(M_train)),
    }


def fitness(
    name: str,
    obj: dict,
    *,
    floor_abs: float = 0.25,
    cov_anchor: float | None = None,
    lambda_floor: float = 3.0,
) -> float:
    """Scalar fitness for `sep`/`cov`/`align`: base value minus `lambda_floor` times the
    retrievability shortfall below `floor_abs`; `align` = -|cov - cov_anchor|, so it peaks
    at the real target instead of minimising without bound.
    """
    if name == "sep":
        base = obj["sep"]
    elif name == "cov":
        base = -obj["cov"]
    elif name == "align":
        if cov_anchor is None:
            raise ValueError("align objective requires cov_anchor")
        base = -abs(obj["cov"] - cov_anchor)
    else:
        raise ValueError(f"unknown objective {name!r}")
    violation = max(0.0, floor_abs - obj["mean_ndcg"])
    return base - lambda_floor * violation


def pick_winners(df: pd.DataFrame, objectives: list[str], floor_abs: float = 0.25,
                 anchor: dict | None = None) -> dict:
    """Best directive per objective among rows with mean nDCG@10 >= `floor_abs` (`sep` max,
    `cov` min, `align` closest to anchor); returns `{name: {"directive", "dirhash",
    "row"}}`.
    """
    anchor = anchor or {}
    cov_anchor = anchor.get("anchor")
    base_row = df[df["directive"] == ""].iloc[0]
    winners = {"base": {"directive": "", "dirhash": base_row["dirhash"],
                        "row": base_row.to_dict()}}
    for obj in objectives:
        g = df[df["objective"] == obj]
        elig = g[g["mean_ndcg"] >= floor_abs]
        if elig.empty:
            elig = g
        if obj == "sep":
            best = elig.loc[elig["sep"].idxmax()]
        elif obj == "cov":
            best = elig.loc[elig["cov"].idxmin()]
        elif obj == "align":
            if cov_anchor is None:
                raise ValueError("align winner needs a real-query anchor")
            best = elig.loc[(elig["cov"] - cov_anchor).abs().idxmin()]
        else:
            raise ValueError(f"unknown objective {obj!r}")
        winners[f"best_{obj}"] = {"directive": best["directive"],
                                  "dirhash": best["dirhash"], "row": best.to_dict()}
    return winners


def transfer(M: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    """Returns (Kendall tau-b, Pearson r) of the pool's per-pipeline mean nDCG@10 vs the real
    target.
    """
    synth = M.mean(axis=0)
    tau = kendalltau(synth, real).statistic
    r = pearsonr(synth, real).statistic if len(synth) > 2 else float("nan")
    return float(tau), float(r)


def family_profile(M_full: np.ndarray, roster: list[str], paradigms: dict) -> dict:
    """Mean nDCG@10 per retrieval paradigm; the label-free signal showing which families the
    queries over-reward.
    """
    means = M_full.mean(axis=0)
    by: dict[str, list[float]] = {}
    for p, m in zip(roster, means):
        by.setdefault(paradigms.get(p, "other"), []).append(float(m))
    return {fam: float(np.mean(v)) for fam, v in by.items()}


@dataclass
class Candidate:
    directive: str
    metrics: dict = field(default_factory=dict)
    origin: str = "seed"
    iteration: int = 0


REFLECT_SYSTEM = (
    "You are optimizing the instruction given to a question-generator. The "
    "generator reads a document and writes one search question a real user might "
    "type. The generated questions are then used to RANK a fixed set of retrieval "
    "pipelines (lexical, learned-sparse, dense, fusion, rerank, late-interaction) "
    "by how well each retrieves the source passage. Your job: propose a better "
    "one-paragraph instruction so the resulting question set is a better, less "
    "biased ranking instrument. Output ONLY the new instruction text, no preamble, "
    "no quotes, no explanation."
)

# OPRO (Yang et al., ICLR 2024) reuses the same persona.
OPRO_SYSTEM = REFLECT_SYSTEM


def _direction_line(objective_name: str, target: float | None, floor_abs: float,
                    archive: list[Candidate]) -> str:
    """One plain-language DIRECTION line naming the retrievability floor or real-query target.
    """
    floor_txt = f"{floor_abs:.2f}"
    if objective_name == "cov":
        return (f"DIRECTION: decrease coverage as far as possible while keeping nDCG@10 "
                f"at or above {floor_txt} - below that the score is penalized.")
    if objective_name == "sep":
        return (f"DIRECTION: increase separation (spread of nDCG across pipelines) "
                f"while keeping nDCG@10 at or above {floor_txt}.")
    if objective_name == "align":
        best_cov = _best_archive_metric(archive, "cov")
        tgt = f"{target:.2f}" if target is not None else "the real level"
        line = (f"DIRECTION: match the target coverage {tgt}. If your coverage is ABOVE "
                f"the target, REDUCE it; if BELOW, INCREASE it.")
        if best_cov is not None:
            line += f" Your best attempt so far reached coverage {best_cov:.2f}."
        return line
    raise ValueError(f"unknown objective {objective_name!r}")


def _best_archive_metric(archive: list[Candidate], key: str) -> float | None:
    """The named metric off the archive's best-fitness-so-far candidate, or None if empty."""
    if not archive:
        return None
    best = max(archive, key=lambda c: c.metrics.get("fitness", float("-inf")))
    return best.metrics.get(key)


def _opro_objective_header(objective_name: str, archive: list[Candidate], *,
                           real_context: dict | None = None,
                           direction_feedback: bool = False,
                           floor_abs: float = 0.25) -> tuple[list[str], float | None]:
    """Builds the OPTIMIZATION TARGET, optional DIRECTION, and HARD CONSTRAINT lines; returns
    `(lines, target)`.
    """
    obj_def = OBJECTIVES[objective_name]
    sense = _SENSE[objective_name]
    lines: list[str] = []
    target = None
    if sense == "match":
        if real_context is not None:
            target = real_context.get("anchor")
        head = f"OPTIMIZATION TARGET (match the real-query level): {obj_def}."
        if target is not None:
            head += f" on a sample of real user questions this quantity is {target:.3f}."
        lines.append(head)
    else:
        lines.append(f"OPTIMIZATION TARGET ({sense}): {obj_def}.")
    lines.append("")
    if direction_feedback:
        lines.append(_direction_line(objective_name, target, floor_abs, archive))
        lines.append("")
    lines.append(
        "HARD CONSTRAINT: every question must stay genuinely answerable from its source "
        "passage — a knowledgeable person holding that passage must agree it answers the "
        "question."
    )
    lines.append("")
    return lines, target


def build_opro_meta_prompt(objective_name: str, archive: list[Candidate], *,
                           real_context: dict | None = None,
                           sample_docs: list[str] | None = None,
                           meta_max_traj: int = 20,
                           direction_feedback: bool = False,
                           floor_abs: float = 0.25) -> str:
    """Renders the OPRO meta-prompt (Yang et al., ICLR 2024): objective definition plus the
    trajectory of tried `(directive, fitness)` pairs, sorted ascending (best last, per
    OPRO's recency-bias finding) and capped to `meta_max_traj`.
    """
    lines, _ = _opro_objective_header(objective_name, archive, real_context=real_context,
                                      direction_feedback=direction_feedback, floor_abs=floor_abs)
    if real_context is not None:
        lines.append("SAMPLE OF REAL USER QUESTIONS:")
        lines += [f"  - {q}" for q in real_context.get("sample", [])[:6]]
        lines.append("")
    if sample_docs:
        lines.append("SAMPLE SOURCE DOCUMENTS (the generator's inputs):")
        for d in sample_docs[:3]:
            lines.append(f"  - {d[:300]}")
        lines.append("")
    lines.append("INSTRUCTIONS TRIED SO FAR AND THEIR SCORES (higher score = better):")
    ordered = sorted(archive, key=lambda c: c.metrics.get("fitness", float("-inf")))
    capped = ordered[-meta_max_traj:] if meta_max_traj else ordered
    for c in capped:
        d = (c.directive or "(none — plain generation)")[:200]
        fit = c.metrics.get("fitness", float("nan"))
        if direction_feedback:
            cov = c.metrics.get("cov", float("nan"))
            ndcg = c.metrics.get("mean_ndcg", float("nan"))
            flag = f"  (nDCG<{floor_abs:.2f} -> penalized)" if ndcg < floor_abs else ""
            lines.append(f"  score={fit:+.3f}  cov={cov:.2f}  nDCG={ndcg:.2f}{flag} | {d}")
        else:
            lines.append(f"  score={fit:+.4f} | {d}")
    lines.append("")
    lines.append(
        "Write a NEW instruction, different from every one above, that will achieve a "
        "HIGHER score. Output only the instruction text, no preamble or quotes."
    )
    return "\n".join(lines)


def optimize_opro(objective_name: str, evaluate_fn, optimizer_llm: BedrockLLM, *,
                  k_per_step: int = 3, steps: int = 6,
                  real_context: dict | None = None,
                  sample_docs: list[str] | None = None,
                  meta_max_traj: int = 20, temperature: float = 1.0,
                  max_tokens: int = 300, log=print,
                  direction_feedback: bool = True,
                  floor_abs: float = 0.25) -> list[Candidate]:
    """OPRO archive loop (Yang et al., ICLR 2024) for one objective; `evaluate_fn(directive) ->
    Candidate` must populate `.metrics` with `fitness`, `sep`, `cov`, `mean_ndcg`,
    `family_profile`, `sample_queries`, `transfer_tau_full`, `transfer_r_test`. Returns the
    full archive.
    """
    archive: list[Candidate] = []
    seed = evaluate_fn("")
    seed.origin, seed.iteration = "seed", 0
    archive.append(seed)
    log(f"  [seed] fit={seed.metrics['fitness']:+.4f} "
        f"τ_full={seed.metrics['transfer_tau_full']:+.3f} "
        f"r_test={seed.metrics['transfer_r_test']:+.3f} "
        f"ndcg={seed.metrics['mean_ndcg']:.3f} cov={seed.metrics['cov']:.3f} :: (base)")

    for step in range(1, steps + 1):
        prompt = build_opro_meta_prompt(objective_name, archive, real_context=real_context,
                                        sample_docs=sample_docs, meta_max_traj=meta_max_traj,
                                        direction_feedback=direction_feedback,
                                        floor_abs=floor_abs)
        seen = {c.directive for c in archive}
        n_new = 0
        for _ in range(k_per_step):
            resp = optimizer_llm.invoke([BedrockLLM.user_message(prompt)], system=OPRO_SYSTEM,
                                        temperature=temperature, max_tokens=max_tokens)
            directive = resp.text.strip().strip('"').strip()
            if not directive or directive in seen:
                continue
            seen.add(directive)
            child = evaluate_fn(directive)
            child.origin, child.iteration = f"opro:{step}", step
            archive.append(child)
            n_new += 1
            log(f"  [step {step}] fit={child.metrics['fitness']:+.4f} "
                f"τ_full={child.metrics['transfer_tau_full']:+.3f} "
                f"r_test={child.metrics['transfer_r_test']:+.3f} "
                f"ndcg={child.metrics['mean_ndcg']:.3f} cov={child.metrics['cov']:.3f} :: "
                f"{directive[:70]}")
        if n_new == 0:
            log(f"  [step {step}] no new candidates (all empty or duplicate)")
    return archive
