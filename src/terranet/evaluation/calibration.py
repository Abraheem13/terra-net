"""UQ calibration: empirical coverage vs nominal, interval width, ECE-for-regression."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gaussian_interval_coverage(mean, sigma, target, alphas=(0.5, 0.8, 0.9, 0.95)):
    rows = []
    for a in alphas:
        z = norm.ppf(0.5 + a / 2)
        lo, hi = mean - z * sigma, mean + z * sigma
        cov = float(((target >= lo) & (target <= hi)).mean())
        rows.append({"nominal": a, "empirical": cov,
                     "avg_width": float((hi - lo).mean()), "gap": cov - a})
    return rows


def regression_ece(mean, sigma, target, n_bins: int = 20) -> float:
    """Expected calibration error over predicted CDF quantiles (Kuleshov et al.)."""
    q = norm.cdf((target - mean) / np.maximum(sigma, 1e-9))
    levels = np.linspace(0, 1, n_bins + 1)[1:]
    emp = np.array([(q <= l).mean() for l in levels])
    return float(np.abs(emp - levels).mean())
