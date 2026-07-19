"""Honest statistics: paired bootstrap CIs over LOCO folds + Holm-Bonferroni.

We deliberately avoid reporting bare Cohen's d on per-measurement data (spatially
correlated samples inflate d to absurd values, cf. base paper's d = 11-15).
Effect sizes are computed on per-fold means, with bootstrap CIs.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                        alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """a, b: per-fold metric values (same folds). Returns diff mean and CI for a-b."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff_mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired t-test AND Wilcoxon over folds; report both."""
    t, pt = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = np.nan, np.nan
    d = a - b
    dz = d.mean() / max(d.std(ddof=1), 1e-12)  # Cohen's dz on fold-level pairs
    return {"t": float(t), "p_t": float(pt), "wilcoxon": float(w), "p_w": float(pw),
            "cohens_dz_folds": float(dz), "n_folds": len(a)}


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop = {}, False
    for i, (k, p) in enumerate(items):
        if not stop and p <= alpha / (m - i):
            out[k] = True
        else:
            stop = True
            out[k] = False
    return out
