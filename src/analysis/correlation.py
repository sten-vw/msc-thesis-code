"""Rank and linear agreement between per-pipeline score vectors.

`rank_pearson`/`fisher_mean` compare two already-computed {pipeline: score}
vectors; `transfer_ip`/`synth_ip` are bootstrap estimators for the scaling
curve that holds a real-query target fixed and grows the size of the
synthetic sample it is read against.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, pearsonr


def rank_pearson(a_scores: dict, b_scores: dict, keys: list[str]) -> tuple[float, float]:
    """(Kendall tau-b, Pearson r) between two {pipeline: score} dicts over `keys`; returns
    (nan, nan) if either is constant over the roster.
    """
    a = [a_scores[p] for p in keys]
    b = [b_scores[p] for p in keys]
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan"), float("nan")
    return float(kendalltau(a, b).statistic), float(pearsonr(a, b)[0])


def fisher_mean(rs) -> float:
    """Fisher-z averaged Pearson r (clipped to avoid an arctanh blow-up at ±1)."""
    rs = [r for r in rs if not np.isnan(r)]
    if not rs:
        return float("nan")
    z = np.mean([np.arctanh(np.clip(r, -0.999, 0.999)) for r in rs])
    return float(np.tanh(z))


def transfer_ip(
    A: np.ndarray, real_full_mean: np.ndarray, n: int, n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Mean Kendall tau (percentile-bootstrap 95% CI) between an n-sized resample of `A` and
    the fixed real-query target, over `n_boot` draws.
    """
    taus = []
    for _ in range(n_boot):
        idx = rng.integers(0, A.shape[0], n)
        taus.append(kendalltau(A[idx].mean(0), real_full_mean).statistic)
    return (float(np.nanmean(taus)), float(np.nanpercentile(taus, 2.5)),
            float(np.nanpercentile(taus, 97.5)))


def synth_ip(A: np.ndarray, n: int, n_boot: int, rng: np.random.Generator) -> float:
    """Mean Kendall tau between two independent n-sized resamples of `A`, over `n_boot` draws.
    """
    taus = []
    for _ in range(n_boot):
        i1 = rng.integers(0, A.shape[0], n)
        i2 = rng.integers(0, A.shape[0], n)
        taus.append(kendalltau(A[i1].mean(0), A[i2].mean(0)).statistic)
    return float(np.nanmean(taus))
