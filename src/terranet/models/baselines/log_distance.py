"""Log-distance path-loss model and per-tile mixed-path parameter estimation.

PL(d) = PL0 + 10*gamma*log10(d/d0). Mixed-path (base paper Alg. 3 / MAPLE):
cumulative loss = sum over intersected tile segments of per-tile contributions;
solved as one linear least-squares system over all measurements (rather than
incremental refinement) — deterministic, exactly reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


def fit_global(d_m: np.ndarray, pl_db: np.ndarray, d0: float = 1.0) -> tuple[float, float]:
    """Grid-search-free closed-form fit of (gamma, PL0) via least squares."""
    x = 10.0 * np.log10(np.maximum(d_m, d0) / d0)
    A = np.c_[x, np.ones_like(x)]
    (gamma, pl0), *_ = np.linalg.lstsq(A, pl_db, rcond=None)
    return float(gamma), float(pl0)


def predict(d_m: np.ndarray, gamma: float, pl0: float, d0: float = 1.0) -> np.ndarray:
    return pl0 + 10.0 * gamma * np.log10(np.maximum(d_m, d0) / d0)


def fit_tiles_mixed_path(
    segments: list[list[tuple[int, float]]], pl_db: np.ndarray, n_tiles: int,
    d0: float = 1.0, damp: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint LS over tile parameters.

    segments[i] = [(tile_idx, seg_len_m), ...] for measurement i (LoS trace split
    across intersected tiles). Unknowns: gamma_t, pl0_t per tile. Model:
        PL_i = sum_t [ pl0_t * w_it + 10*gamma_t*log10(max(L_it, d0)) ]
    where w_it = L_it / sum_t L_it apportions the reference loss.
    Returns (gamma[n_tiles], pl0[n_tiles], n_meas_per_tile).
    """
    n = len(segments)
    A = lil_matrix((n, 2 * n_tiles))
    counts = np.zeros(n_tiles)
    for i, segs in enumerate(segments):
        total = sum(l for _, l in segs)
        for t, l in segs:
            A[i, t] = 10.0 * np.log10(max(l, d0) / d0)   # gamma_t coefficient
            A[i, n_tiles + t] = l / max(total, d0)        # pl0_t coefficient
            counts[t] += 1
    sol = lsqr(A.tocsr(), pl_db, damp=damp)[0]
    return sol[:n_tiles], sol[n_tiles:], counts
