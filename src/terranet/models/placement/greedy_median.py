"""Base paper's iterative median-based BS planning (Algorithm 6). Baseline."""
from __future__ import annotations

import numpy as np


def greedy_median_placement(points_xy: np.ndarray, candidates_xy: np.ndarray,
                            coverage_fn, p_th: float, max_iter: int = 1000):
    """coverage_fn(bs_xy, pts_xy) -> received power dBm per point.
    Returns indices of selected candidates (existing first, new stations appended)."""
    covered = np.zeros(len(points_xy), dtype=bool)
    selected: list[int] = []
    new_stations: list[np.ndarray] = []
    it = 0
    while not covered.all() and it < max_iter:
        it += 1
        unc = points_xy[~covered]
        med = np.median(unc, axis=0)
        p_star = unc[np.argmin(np.linalg.norm(unc - med, axis=1))]
        best, best_pow = -1, -np.inf
        for j in range(len(candidates_xy)):
            if j in selected:
                continue
            pw = coverage_fn(candidates_xy[j], p_star[None])[0]
            if pw >= p_th and pw > best_pow:
                best, best_pow = j, pw
        if best >= 0:
            selected.append(best)
            bs = candidates_xy[best]
        else:
            new_stations.append(med)
            bs = med
        covered |= coverage_fn(bs, points_xy) >= p_th
    return selected, np.array(new_stations), covered
