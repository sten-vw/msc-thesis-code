"""Bootstrap estimator for the within-real ranking-agreement ceiling.

Kendall tau under query resampling of the real-query per-pipeline scores: the
upper bound synthetic-query ranking agreement is read against, since no
synthetic query set can agree with the real ranking more than the real
ranking agrees with an independent resample of itself.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def within_real_ceiling(
    arr: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000,
) -> dict:
    """`arr` is (Q, R) per-query roster scores; returns mean and 95% CI for `stability_vs_full`
    (resampled vs full-pool mean) and `independent_pairs` (two independent resamples).
    """
    Q = arr.shape[0]
    full = arr.mean(axis=0)
    i1 = rng.integers(0, Q, size=(n_boot, Q))
    i2 = rng.integers(0, Q, size=(n_boot, Q))
    m1 = arr[i1].mean(axis=1)
    m2 = arr[i2].mean(axis=1)
    tA = np.array([kendalltau(full, m1[b]).statistic for b in range(n_boot)])
    tB = np.array([kendalltau(m1[b], m2[b]).statistic for b in range(n_boot)])
    return {
        "stability_vs_full": {
            "tau_mean": float(np.nanmean(tA)),
            "ci_low": float(np.nanpercentile(tA, 2.5)),
            "ci_high": float(np.nanpercentile(tA, 97.5)),
        },
        "independent_pairs": {
            "tau_mean": float(np.nanmean(tB)),
            "ci_low": float(np.nanpercentile(tB, 2.5)),
            "ci_high": float(np.nanpercentile(tB, 97.5)),
        },
    }
