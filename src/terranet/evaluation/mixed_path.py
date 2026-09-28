"""Mixed-path tile models (the path-loss composition of MAPLE and of the
descriptor-based framework it was extended into).

The straight horizontal segment from a site to a pixel is split at the tile
borders; tile t holds a length L_t of it, covering the distances from d_in,t
to d_out,t from the site. Three compositions are evaluated:

  sum        PL = sum_t [A_t + 10 gamma_t log10(max(L_t, 1 m))]
             the cumulative loss as written in the published framework
             (Algorithm 3, line 11, reference distance 1 m): every crossed
             tile adds its own log-distance loss over its own segment;
  piecewise  PL = A_s + sum_t gamma_t 10 log10(max(d_out,t, d0) / max(d_in,t, d0))
             a multi-slope law along the path: the intercept of the site's
             tile s at d0 and, over each stretch of distance, the exponent of
             the tile it lies in (the loss is continuous in distance);
  average    PL = sum_t (L_t / d_h) [A_t + gamma_t x],  x = 10 log10(d / d0)
             path-averaged parameters (length-weighted mixed-path
             interpolation).

With the whole path in one tile, piecewise and average reduce to the
log-distance law of that tile. All three are linear in the tile parameters
theta_t = (A_t, gamma_t), so the tile labels of a city are fitted jointly by
least squares over all of its links, with a weak ridge penalty towards the
best city-wide parameter pair so that tiles crossed by few links (and, for
piecewise, the intercepts of tiles without a site) stay identifiable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from terranet.data.tiling import point_tile_index

MODELS = ("sum", "piecewise", "average")
STEP_M = 10.0          # sampling step along the segment (the pixel size)
RIDGE = 10.0           # penalty towards the city-wide pair (pseudo-links)


@dataclass
class Paths:
    """Per (pixel, site) pair: segment length (L) and log-distance increment
    (Q, dB per unit exponent) in every crossed tile, and the site's tile."""
    L: sp.csr_matrix
    Q: sp.csr_matrix
    first: np.ndarray

    def rows(self, mask):
        return Paths(self.L[mask], self.Q[mask], self.first[mask])

    @property
    def shape(self):
        return self.L.shape


def segment_lengths(city, grid, frame, d0=100.0, step=STEP_M, chunk=4000):
    """Paths of every (pixel, site) pair of `city` over the tiles of `grid`.

    Rows are pairs k = p * S + s; columns are rows of the tile grid. The
    horizontal segment is divided into n equal pieces of about `step` metres;
    piece j covers the distances [j, j + 1] d_h / n from the site and is
    assigned to the tile of its midpoint."""
    res = step / 2
    x0 = min(city.px.min(), city.sx.min()) - step
    y0 = min(city.py.min(), city.sy.min()) - step
    nx = int(np.ceil((max(city.px.max(), city.sx.max()) + step - x0) / res)) + 1
    ny = int(np.ceil((max(city.py.max(), city.sy.max()) + step - y0) / res)) + 1
    gx, gy = np.meshgrid(x0 + res * np.arange(nx), y0 + res * np.arange(ny), indexing="ij")
    lat, lon = frame.to_geo(gx.ravel(), gy.ravel())
    R = point_tile_index(grid, lon, lat).reshape(nx, ny)

    P, S, T = len(city.px), len(city.sx), len(grid)
    dx = city.px[:, None] - city.sx[None, :]
    dy = city.py[:, None] - city.sy[None, :]
    dh = np.hypot(dx, dy).ravel()
    n = np.maximum(np.ceil(dh / step).astype(np.int64), 1)
    sxk, syk = np.tile(city.sx, P), np.tile(city.sy, P)
    dxk, dyk = dx.ravel(), dy.ravel()
    first = np.full(P * S, -1, np.int64)
    rows, cols, lens, incs = [], [], [], []
    for a in range(0, P * S, chunk):
        b = min(a + chunk, P * S)
        k = np.arange(a, b)
        nk = n[a:b]
        j = np.arange(int(nk.max()))[None, :]
        valid = j < nk[:, None]
        f = (j + 0.5) / nk[:, None]
        X = sxk[a:b, None] + f * dxk[a:b, None]
        Y = syk[a:b, None] + f * dyk[a:b, None]
        ix = np.clip(np.rint((X - x0) / res).astype(np.int64), 0, nx - 1)
        iy = np.clip(np.rint((Y - y0) / res).astype(np.int64), 0, ny - 1)
        t = R[ix, iy]
        piece = dh[a:b, None] / nk[:, None]
        lo = np.maximum(j * piece, d0)
        hi = np.maximum((j + 1) * piece, d0)
        inc = 10 * np.log10(hi / lo)
        keep = valid & (t >= 0)
        kk = np.broadcast_to(k[:, None], t.shape)[keep]
        key, inv = np.unique(kk * T + t[keep], return_inverse=True)
        rows.append(key // T)
        cols.append(key % T)
        lens.append(np.bincount(inv, weights=np.broadcast_to(piece, t.shape)[keep]))
        incs.append(np.bincount(inv, weights=inc[keep]))
        t_first = np.where(keep, t, -1)
        has = keep.any(1)
        first[a:b][has] = t_first[has, keep[has].argmax(1)]
    r, c = np.concatenate(rows), np.concatenate(cols)
    L = sp.csr_matrix((np.concatenate(lens).astype(np.float32), (r, c)), shape=(P * S, T))
    Q = sp.csr_matrix((np.concatenate(incs).astype(np.float32), (r, c)), shape=(P * S, T))
    return Paths(L, Q, first)


def design(model, paths, x):
    """Design matrices (X_A, X_gamma) of a composition; x: 10 log10(d / d0)
    per pair."""
    L = paths.L.tocsr()
    T = L.shape[1]
    if model == "sum":
        XA = L.copy()
        XA.data = np.ones_like(XA.data)
        XG = L.copy()
        XG.data = 10 * np.log10(np.maximum(XG.data, 1.0))
    elif model == "piecewise":
        ok = paths.first >= 0
        XA = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), paths.first[ok])),
                           shape=(L.shape[0], T))
        XG = paths.Q.tocsr()
    elif model == "average":
        tot = np.asarray(L.sum(1)).ravel()
        W = sp.diags(1 / np.maximum(tot, 1e-9)) @ L
        XA = W.tocsr()
        XG = (sp.diags(x) @ W).tocsr()
    else:
        raise ValueError(model)
    return XA, XG


def predict(model, paths, x, A, G):
    XA, XG = design(model, paths, x)
    return XA @ A + XG @ G


def fit(model, paths, x, y, ridge=RIDGE):
    """Joint least-squares tile parameters. Returns (A, gamma, n_links) per
    tile, n_links being the number of fitted links crossing the tile."""
    XA, XG = design(model, paths, x)
    # city-wide pair: the same (A, gamma) in every tile
    g = np.column_stack([np.asarray(XA.sum(1)).ravel(), np.asarray(XG.sum(1)).ravel()])
    a0, g0 = np.linalg.lstsq(g, y, rcond=None)[0]
    X = sp.hstack([XA, XG]).tocsr()
    r = y - (a0 * g[:, 0] + g0 * g[:, 1])
    d = lsqr(X, r, damp=np.sqrt(ridge), atol=1e-8, btol=1e-8, iter_lim=2000)[0]
    T = XA.shape[1]
    n_links = np.asarray((paths.L > 0).sum(0)).ravel()
    return a0 + d[:T], g0 + d[T:], n_links
