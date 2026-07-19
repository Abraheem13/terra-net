"""Submodular coverage maximization with lazy greedy — (1-1/e) guarantee.

Objective F(S) = # of high-priority points covered by at least one BS in S
(monotone submodular). Budgeted variant: pick k stations maximizing coverage;
or cost-minimal cover via greedy set-cover (ln n approximation). Operates on the
LEARNED, uncertainty-aware coverage: a point counts as covered if the conformal
LOWER bound of received power clears the threshold (risk-averse planning).
"""
from __future__ import annotations

import heapq

import numpy as np


def lazy_greedy_max_coverage(cover_matrix: np.ndarray, k: int) -> tuple[list[int], np.ndarray]:
    """cover_matrix: (n_candidates, n_points) boolean. Returns (selected, gain_curve).

    Lazy evaluation: entries are stamped with the selection count at which their
    gain was computed; a popped entry is only trusted if its stamp is current.
    """
    n_cand = cover_matrix.shape[0]
    covered = np.zeros(cover_matrix.shape[1], dtype=bool)
    heap = [(-int(cover_matrix[j].sum()), 0, j) for j in range(n_cand)]
    heapq.heapify(heap)
    selected: list[int] = []
    gains: list[int] = []
    while heap and len(selected) < k:
        neg_gain, stamp, j = heapq.heappop(heap)
        if stamp == len(selected):          # bound is current -> safe to select
            if -neg_gain <= 0:
                break                       # no candidate adds coverage
            selected.append(j)
            gains.append(-neg_gain)
            covered |= cover_matrix[j]
        else:                               # stale -> recompute, push back
            g = int((cover_matrix[j] & ~covered).sum())
            heapq.heappush(heap, (-g, len(selected), j))
    return selected, np.array(gains)


def greedy_set_cover(cover_matrix: np.ndarray, required: np.ndarray | None = None) -> list[int]:
    """Minimal-cardinality cover of all points (ln n approx)."""
    need = np.ones(cover_matrix.shape[1], bool) if required is None else required.copy()
    selected: list[int] = []
    while need.any():
        gains = (cover_matrix & need).sum(1)
        j = int(np.argmax(gains))
        if gains[j] == 0:
            break  # infeasible points remain
        selected.append(j)
        need &= ~cover_matrix[j]
    return selected


def risk_averse_cover_matrix(power_lower_bound_dbm: np.ndarray, p_th: float) -> np.ndarray:
    """power_lower_bound_dbm: (n_candidates, n_points) conformal lower bounds."""
    return power_lower_bound_dbm >= p_th
