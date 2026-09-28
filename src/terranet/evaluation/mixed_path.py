"""Mixed-path tile models (the path-loss composition of MAPLE and of the
descriptor-based framework it was extended into).

The straight horizontal segment from a site to a pixel is split at the tile
borders; tile t holds a length L_t of it. Two compositions are evaluated:

  sum      PL = sum_t [A_t + 10 gamma_t log10(max(L_t, 1 m))]
           the cumulative loss of the published framework (Algorithm 3, line 11,
           reference distance 1 m): every crossed tile adds its own
           log-distance loss over its own segment;
  average  PL = sum_t (L_t / d_h) [A_t + gamma_t x],  x = 10 log10(d / d0)
           path-averaged parameters (length-weighted mixed-path interpolation),
           which reduces to the receiver-tile model when the whole segment
           lies in one tile.

Both are linear in the tile parameters theta_t = (A_t, gamma_t), so the tile
labels of a city are fitted jointly by least squares over all of its links,
with a weak ridge penalty towards the best city-wide parameter pair so that
tiles crossed by few links stay identifiable.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from terranet.data.tiling import point_tile_index

MODELS = ("sum", "average")
STEP_M = 10.0          # sampling step along the segment (the pixel size)
RIDGE = 10.0           # penalty towards the city-wide pair (pseudo-links)


def segment_lengths(city, grid, frame, step=STEP_M, chunk=4000):
    """Sparse (pixel*site) x tile matrix of segment lengths (m).

    Rows are pairs k = p * S + s over every pixel p and site s of `city`;
    columns are rows of the tile grid. Points are sampled every `step` metres
    along the horizontal segment (midpoints of equal sub-segments), so each
    sample carries d_h / n of length."""
    # tile raster in the local frame (one lookup per 5 m cell)
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
    sxk = np.tile(city.sx, P)
    syk = np.tile(city.sy, P)
    dxk, dyk = dx.ravel(), dy.ravel()
    rows, cols, vals = [], [], []
    for a in range(0, P * S, chunk):
        b = min(a + chunk, P * S)
        k = np.arange(a, b)
        nk = n[a:b]
        m = int(nk.max())
        j = np.arange(m)[None, :]
        valid = j < nk[:, None]
        f = (j + 0.5) / nk[:, None]
        X = sxk[a:b, None] + f * dxk[a:b, None]
        Y = syk[a:b, None] + f * dyk[a:b, None]
        ix = np.clip(np.rint((X - x0) / res).astype(np.int64), 0, nx - 1)
        iy = np.clip(np.rint((Y - y0) / res).astype(np.int64), 0, ny - 1)
        t = R[ix, iy]
        keep = valid & (t >= 0)
        kk = np.broadcast_to(k[:, None], t.shape)[keep]
        key, cnt = np.unique(kk * T + t[keep], return_counts=True)
        rows.append(key // T)
        cols.append(key % T)
        vals.append(cnt * (dh[key // T] / n[key // T]))
    L = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(P * S, T))
    return L


def design(model, L, x):
    """Design matrices (X_A, X_gamma) of a composition for the pairs of L;
    x: 10 log10(d / d0) per pair."""
    L = L.tocsr()
    if model == "sum":
        XA = L.copy()
        XA.data = np.ones_like(XA.data)
        XG = L.copy()
        XG.data = 10 * np.log10(np.maximum(XG.data, 1.0))
    elif model == "average":
        tot = np.asarray(L.sum(1)).ravel()
        W = sp.diags(1 / np.maximum(tot, 1e-9)) @ L
        XA = W.tocsr()
        XG = (sp.diags(x) @ W).tocsr()
    else:
        raise ValueError(model)
    return XA, XG


def predict(model, L, x, A, G):
    XA, XG = design(model, L, x)
    return XA @ A + XG @ G


def fit(model, L, x, y, ridge=RIDGE):
    """Joint least-squares tile parameters. Returns (A, gamma, n_links) per
    tile, n_links being the number of fitted links crossing the tile."""
    XA, XG = design(model, L, x)
    # city-wide pair: the same (A, gamma) in every tile
    g = np.column_stack([np.asarray(XA.sum(1)).ravel(), np.asarray(XG.sum(1)).ravel()])
    a0, g0 = np.linalg.lstsq(g, y, rcond=None)[0]
    X = sp.hstack([XA, XG]).tocsr()
    r = y - (a0 * g[:, 0] + g0 * g[:, 1])
    d = lsqr(X, r, damp=np.sqrt(ridge), atol=1e-8, btol=1e-8, iter_lim=2000)[0]
    T = L.shape[1]
    n_links = np.asarray((L > 0).sum(0)).ravel()
    return a0 + d[:T], g0 + d[T:], n_links
