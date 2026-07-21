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


def fit_tiles_local(tile_idx: np.ndarray, d_m: np.ndarray, pl_db: np.ndarray,
                    n_tiles: int, min_count: int = 10, d0: float = 1.0):
    """Well-conditioned per-tile fit: each tile uses only measurements whose RX
    lies inside it, with full Tx-Rx distance. Identifiable with a single BS,
    unlike the joint mixed-path system. Returns (gamma, pl0, counts, rmse_db)."""
    gamma = np.full(n_tiles, np.nan)
    pl0 = np.full(n_tiles, np.nan)
    rmse = np.full(n_tiles, np.nan)
    counts = np.bincount(tile_idx, minlength=n_tiles).astype(float)
    for t in range(n_tiles):
        m = tile_idx == t
        if m.sum() < min_count:
            continue
        g, p = fit_global(d_m[m], pl_db[m], d0=d0)
        gamma[t], pl0[t] = g, p
        res = pl_db[m] - predict(d_m[m], g, p, d0=d0)
        rmse[t] = float(np.sqrt((res ** 2).mean()))
    return gamma, pl0, counts, rmse


def fit_tiles_local_ridge(tile_idx: np.ndarray, d_m: np.ndarray, pl_db: np.ndarray,
                          n_tiles: int, min_count: int = 10, d0: float = 1.0,
                          n_prior: float = 50.0):
    """Per-tile fit with Tikhonov shrinkage toward the GLOBAL (gamma, pl0).

    Solves (A^T A + lam*I) theta = A^T y + lam*theta_global with lam = n_prior,
    i.e. the global fit acts as ~n_prior pseudo-measurements. Tiles with wide
    log-distance spans override the prior; narrow-span tiles (unidentifiable
    slope) shrink to global instead of exploding. Returns
    (gamma, pl0, counts, rmse_db, shrinkage in [0,1], theta_global)."""
    g_glob, p_glob = fit_global(d_m, pl_db, d0=d0)
    theta_g = np.array([g_glob, p_glob])
    gamma = np.full(n_tiles, np.nan)
    pl0 = np.full(n_tiles, np.nan)
    rmse = np.full(n_tiles, np.nan)
    shrink = np.full(n_tiles, np.nan)
    counts = np.bincount(tile_idx, minlength=n_tiles).astype(float)
    for t in range(n_tiles):
        m = tile_idx == t
        n = int(m.sum())
        if n < min_count:
            continue
        x = 10.0 * np.log10(np.maximum(d_m[m], d0) / d0)
        A = np.c_[x, np.ones(n)]
        lam = n_prior
        lhs = A.T @ A + lam * np.eye(2)
        rhs = A.T @ pl_db[m] + lam * theta_g
        theta = np.linalg.solve(lhs, rhs)
        gamma[t], pl0[t] = theta
        res = pl_db[m] - A @ theta
        rmse[t] = float(np.sqrt((res ** 2).mean()))
        shrink[t] = lam / (lam + n * max(np.var(x), 1e-9))
    return gamma, pl0, counts, rmse, shrink, theta_g
