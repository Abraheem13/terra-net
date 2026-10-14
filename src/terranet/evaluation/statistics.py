"""City-level inference helpers: the Holm-Bonferroni step-down procedure used
by scripts/10_stats.py (tiles of one city are spatially correlated, so the
unit of inference is the city).
"""
from __future__ import annotations


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
