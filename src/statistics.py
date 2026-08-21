from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import kruskal, spearmanr
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class KruskalResult:
    feature: str
    statistic: float
    p_value: float
    q_value: float


def kruskal_by_feature(X, labels, feature_names: Iterable[str]) -> list[KruskalResult]:
    """Kruskal-Wallis tests for multiple features with Benjamini-Hochberg FDR."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    names = list(feature_names)
    groups = np.unique(labels)

    raw = []
    for j, name in enumerate(names):
        samples = [X[labels == g, j] for g in groups]
        samples = [s[np.isfinite(s)] for s in samples]
        if sum(len(s) > 0 for s in samples) < 2:
            stat, p = np.nan, np.nan
        else:
            stat, p = kruskal(*samples, nan_policy="omit")
        raw.append((name, float(stat), float(p)))

    pvals = np.array([r[2] for r in raw], dtype=float)
    finite = np.isfinite(pvals)
    qvals = np.full_like(pvals, np.nan)
    if finite.any():
        qvals[finite] = multipletests(pvals[finite], method="fdr_bh")[1]

    return [KruskalResult(name, stat, p, float(q)) for (name, stat, p), q in zip(raw, qvals)]


def bootstrap_mean_ci(values, n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(x, size=x.size, replace=True))
    alpha = (1.0 - ci) / 2.0
    return float(np.mean(x)), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def rank_stability(x, y) -> tuple[float, float]:
    """Spearman rank correlation for the same descriptor at two resolutions/scales."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)
