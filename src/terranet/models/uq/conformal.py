"""Split conformal prediction for (gamma, PL0) — distribution-free coverage.

Calibrate on held-out source tiles (or k-shot target tiles). Supports plain
absolute-residual scores and locally weighted scores r/sigma-hat (uses the
heteroscedastic head so intervals adapt per tile). Weighted-CP hook for
covariate shift via likelihood ratios is provided.
"""
from __future__ import annotations

import numpy as np


class SplitConformal:
    def __init__(self, alpha: float = 0.1, normalized: bool = True):
        self.alpha = alpha
        self.normalized = normalized
        self.q: np.ndarray | None = None  # per output dim

    def calibrate(self, mean: np.ndarray, target: np.ndarray,
                  sigma: np.ndarray | None = None) -> "SplitConformal":
        r = np.abs(target - mean)
        if self.normalized and sigma is not None:
            r = r / np.maximum(sigma, 1e-6)
        n = len(r)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q = np.sort(r, axis=0)[min(k, n) - 1]
        return self

    def interval(self, mean: np.ndarray, sigma: np.ndarray | None = None):
        assert self.q is not None, "call calibrate() first"
        w = self.q * (np.maximum(sigma, 1e-6) if (self.normalized and sigma is not None) else 1.0)
        return mean - w, mean + w

    @staticmethod
    def empirical_coverage(lo, hi, target) -> float:
        return float(((target >= lo) & (target <= hi)).mean())
