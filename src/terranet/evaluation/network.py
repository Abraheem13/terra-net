"""Network-level evaluation of predicted path-loss maps against ray tracing.

A `City` holds, for one city, every receiver cell of the solver grid that lies
inside a labelled tile (including the cells that no site reaches below the
cut-off, which are coverage holes), every site, the true path-loss matrix and
the pixel-site geometry. Any predictor that produces a (pixel x site) path-loss
matrix (inf = predicted beyond the cut-off) is scored with `City.evaluate`
(path loss, link existence, serving site, coverage, SINR, spectral efficiency,
cell edges) and `City.plan` (greedy site selection scored on the truth).

Link budget (downlink, full buffer, all sites active): P_tx = 46 dBm,
20 MHz, noise figure 7 dB -> N = -94.0 dBm. Pairs absent from the ray-traced
data exceeded the 150 dB cut-off (received power 10 dB below the noise floor)
and are treated as zero received power; a pixel that no site reaches is an
outage pixel of the truth (uncovered at every level, SINR at the floor).
Spectral efficiency: 3GPP TR 36.942 attenuated Shannon, 0.6 log2(1 + SINR),
zero below -10 dB, capped at 4.4 b/s/Hz. SINR below -20 dB, including pixels
where no site is received, is set to -20 dB (outage) for the truth and every
predictor alike. A pixel at which a predictor receives no site has no
predicted serving site; it counts as a wrong association, as uncovered and as
an outage.

Metrics can be restricted to a pixel mask (`mask`), which the few-shot
experiments use to score the unsurveyed pixels only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from terranet.data.labels import link_distance_m, load_links, log_distance
from terranet.data.sionna_gen.geo import LocalFrame
from terranet.data.tiling import grid_frame, point_tile_index, region_from_points
from terranet.models.placement.submodular import lazy_greedy_max_coverage

P_TX_DBM, NOISE_DBM = 46.0, -174.0 + 10 * np.log10(20e6) + 7.0
CUTOFF_DB = 150.0          # coupling-loss cut-off of the corpus (censoring level)
COV_LEVELS = (120.0, 130.0, 140.0)
PLAN_LEVEL = 140.0
PLAN_K = (3, 5, 8)
COV_TARGET = 0.90          # sites needed to cover this fraction at PLAN_LEVEL
SINR_FLOOR_DB = -20.0      # outage: lower SINR (or no received site) counts as this value


def se_tr36942(sinr_db):
    s = 0.6 * np.log2(1 + 10 ** (sinr_db / 10))
    return np.where(sinr_db < -10, 0.0, np.minimum(s, 4.4))


def _lattice(v, cell):
    """Regular 1-D lattice (origin, count) covering the sorted unique values v."""
    lo, hi = float(v.min()), float(v.max())
    n = int(round((hi - lo) / cell)) + 1
    return lo, n


class City:
    """Pixel x site geometry and the ray-traced truth for one city."""

    def __init__(self, meas, grid, tiles, d0, frame):
        pix, pix_id = np.unique(np.c_[meas.rx_lat.to_numpy(), meas.rx_lon.to_numpy()],
                                axis=0, return_inverse=True)
        pix_id = pix_id.ravel()
        sites, site_id = np.unique(meas.bs_id.to_numpy(), return_inverse=True)
        site_id = site_id.ravel()
        S = len(sites)
        rx_h = float(meas.rx_h.iloc[0])
        # observed receiver cells in the scene's own metric frame (as the solver)
        ox, oy = frame.to_local(pix[:, 1], pix[:, 0])
        ox, oy = np.asarray(ox, float), np.asarray(oy, float)
        cell = float(np.median(np.diff(np.unique(np.round(ox, 3)))))
        self.cell = cell
        # the complete solver lattice spanned by the observed cells (holes included)
        x0, nx = _lattice(ox, cell)
        y0, ny = _lattice(oy, cell)
        oi = np.rint((ox - x0) / cell).astype(int)
        oj = np.rint((oy - y0) / cell).astype(int)
        err = max(np.abs(ox - (x0 + oi * cell)).max(), np.abs(oy - (y0 + oj * cell)).max())
        if err > 0.05 * cell:
            raise ValueError(f"observed cells are off the lattice (max offset {err:.3f} m)")
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        gi, gj = gi.ravel(), gj.ravel()
        gx, gy = x0 + gi * cell, y0 + gj * cell
        glat, glon = frame.to_geo(gx, gy)
        tile = point_tile_index(grid, np.asarray(glon), np.asarray(glat))
        fitted = np.zeros(len(grid), bool)
        fitted[tiles.index[np.isfinite(tiles.gamma)]] = True
        keep = (tile >= 0) & fitted[np.maximum(tile, 0)]
        # lattice (i, j) -> pixel row
        lut = np.full(nx * ny, -1, np.int64)
        lut[np.flatnonzero(keep)] = np.arange(keep.sum())
        self.tile = tile[keep]
        self.px, self.py, self.rx_h = gx[keep], gy[keep], rx_h
        self.ix, self.iy = gi[keep] - gi[keep].min(), gj[keep] - gj[keep].min()
        P = keep.sum()
        # true path loss (inf where the pair was cut off or the cell never reached)
        T = np.full((P, S), np.inf)
        row = lut[oi * ny + oj][pix_id]          # pixel row of every measurement
        lk = row >= 0
        T[row[lk], site_id[lk]] = meas.pathloss_db.to_numpy(float)[lk]
        self.true_pl = T
        self.n_unreached = int((~np.isfinite(T).any(1)).sum())
        # sites
        tx = meas.groupby(site_id)[["tx_lat", "tx_lon", "tx_h"]].first().to_numpy()
        sx, sy = frame.to_local(tx[:, 1], tx[:, 0])
        self.sx, self.sy, self.sz = np.asarray(sx, float), np.asarray(sy, float), tx[:, 2]
        self.sites = sites
        dx = self.px[:, None] - self.sx[None, :]
        dy = self.py[:, None] - self.sy[None, :]
        dz = rx_h - self.sz[None, :]
        self.x = log_distance(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), d0)
        # consistency with the solver distance on observed links
        d_obs = link_distance_m(meas)[lk]
        d_rec = 10 ** (self.x[row[lk], site_id[lk]] / 10) * d0
        self.dist_rel_err = float(np.max(np.abs(d_rec / d_obs - 1)))
        self.obs = np.isfinite(T)
        self.reached = self.obs.any(1)
        self.truth = self.summary(T)

    # ------------------------------------------------------------------
    def summary(self, PL):
        rx = P_TX_DBM - PL                         # dBm, -inf where cut off
        lin = np.where(np.isfinite(rx), 10 ** (rx / 10), 0.0)
        any_ = lin.any(1)
        serv = np.where(any_, np.argmax(lin, 1), -1)
        s = lin[np.arange(len(lin)), np.maximum(serv, 0)] * any_
        with np.errstate(divide="ignore", invalid="ignore"):
            sinr = 10 * np.log10(np.maximum(s, 1e-30) / (lin.sum(1) - s + 10 ** (NOISE_DBM / 10)))
        sinr = np.maximum(sinr, SINR_FLOOR_DB)
        serv_pl = np.where(any_, PL[np.arange(len(PL)), np.maximum(serv, 0)], np.inf)
        return dict(serv=serv, serv_pl=serv_pl, sinr=sinr, se=se_tr36942(sinr))

    def subset(self, site_idx):
        """The same city served by a subset of the sites (pixels unchanged)."""
        c = copy.copy(self)
        site_idx = np.asarray(site_idx)
        c.true_pl, c.x = self.true_pl[:, site_idx], self.x[:, site_idx]
        c.sx, c.sy, c.sz = self.sx[site_idx], self.sy[site_idx], self.sz[site_idx]
        c.sites, c.obs = self.sites[site_idx], self.obs[:, site_idx]
        c.reached = c.obs.any(1)
        c.n_unreached = int((~c.reached).sum())
        c.truth = c.summary(c.true_pl)
        return c

    def _boundary(self, serv):
        """Boolean raster of cell-edge pixels: the pixel's serving site differs
        from that of its east or north neighbour (one pixel per boundary
        crossing, so edges are one pixel thick). Pixels without a serving
        site (-1) do not form edges."""
        nx, ny = self.ix.max() + 1, self.iy.max() + 1
        A = np.full((nx, ny), -1)
        A[self.ix, self.iy] = serv
        B = np.zeros_like(A, bool)
        B[:-1] |= (A[1:] != A[:-1]) & (A[1:] >= 0) & (A[:-1] >= 0)
        B[:, :-1] |= (A[:, 1:] != A[:, :-1]) & (A[:, 1:] >= 0) & (A[:, :-1] >= 0)
        return B

    def _mask_raster(self, mask):
        nx, ny = self.ix.max() + 1, self.iy.max() + 1
        M = np.zeros((nx, ny), bool)
        M[self.ix[mask], self.iy[mask]] = True
        return M

    def boundary_displacement(self, serv_true, serv_pred, mask=None):
        """Mean distance (m) between true and predicted cell edges, averaged
        over both directions (symmetric mean chamfer distance), over the edge
        pixels inside `mask`."""
        bt, bp = self._boundary(serv_true), self._boundary(serv_pred)
        if mask is not None:
            M = self._mask_raster(mask)
            bt_m, bp_m = bt & M, bp & M
        else:
            bt_m, bp_m = bt, bp
        if not bt_m.any() or not bp_m.any() or not bt.any() or not bp.any():
            return float("nan")
        dt = distance_transform_edt(~bp, sampling=self.cell)[bt_m].mean()
        dp = distance_transform_edt(~bt, sampling=self.cell)[bp_m].mean()
        return float((dt + dp) / 2)

    def serving_residual(self, PLh):
        """Per pixel: predicted serving path loss, the residual r = true minus
        predicted serving path loss, and the true serving path loss (inf for a
        pixel that no site reaches). Coverage at level L holds exactly when
        predicted + r <= L."""
        h = self.summary(PLh)
        with np.errstate(invalid="ignore"):
            r = self.truth["serv_pl"] - h["serv_pl"]
        return h["serv_pl"], r, self.truth["serv_pl"]

    def predicted_pl(self, gamma_t, pl0_t):
        return pl0_t[self.tile][:, None] + gamma_t[self.tile][:, None] * self.x

    # ------------------------------------------------------------------
    def evaluate(self, PLh, pl_for_error=None, mask=None):
        """Network metrics of the predicted map PLh (inf = predicted beyond the
        cut-off), over the pixels in `mask` (default: all). The path-loss error
        over the ray-traced links is computed from `pl_for_error` when given
        (the regression part of a hurdle model); `pl_rmse_cens_db` scores every
        (pixel, site) pair with prediction and truth clipped at the cut-off."""
        m = np.ones(len(self.true_pl), bool) if mask is None else np.asarray(mask, bool)
        t, h = self.truth, self.summary(PLh)
        obs = self.obs & m[:, None]
        with np.errstate(invalid="ignore"):
            E = (PLh if pl_for_error is None else pl_for_error) - self.true_pl
        r = {"n_pixels": int(m.sum()), "n_unreached": int((~self.reached & m).sum()),
             "pl_rmse_db": float(np.sqrt((E[obs] ** 2).mean())),
             "pl_bias_db": float(E[obs].mean())}
        Ec = np.minimum(PLh, CUTOFF_DB) - np.minimum(self.true_pl, CUTOFF_DB)
        r["pl_rmse_cens_db"] = float(np.sqrt((Ec[m] ** 2).mean()))
        # link existence (below the cut-off) and reach of the pixels
        po = np.isfinite(PLh) & m[:, None]
        r["link_precision"] = float((po & obs).sum() / max(po.sum(), 1))
        r["link_recall"] = float((po & obs).sum() / max(obs.sum(), 1))
        pr = np.isfinite(PLh).any(1)
        r["reach_acc"] = float((pr[m] == self.reached[m]).mean())
        # serving site over the pixels the truth reaches; a pixel the predictor
        # does not reach has serving site -1 and counts as wrong
        rm = m & self.reached
        r["assoc_acc"] = float((h["serv"][rm] == t["serv"][rm]).mean())
        for L in COV_LEVELS:
            ct, ch = t["serv_pl"][m] <= L, h["serv_pl"][m] <= L
            r[f"cov_true_{int(L)}"] = float(ct.mean())
            r[f"cov_acc_{int(L)}"] = float((ct == ch).mean())
            r[f"cov_err_{int(L)}"] = float(ch.mean() - ct.mean())
        r["sinr_mae_db"] = float(np.abs(h["sinr"][m] - t["sinr"][m]).mean())
        a, b = np.sort(t["sinr"][m]), np.sort(h["sinr"][m])
        grid = np.linspace(min(a[0], b[0]), max(a[-1], b[-1]), 2000)
        r["sinr_ks"] = float(np.abs(np.searchsorted(a, grid, "right") / len(a)
                                    - np.searchsorted(b, grid, "right") / len(b)).max())
        r["se_true"] = float(t["se"][m].mean())
        r["se_err"] = float(h["se"][m].mean() - t["se"][m].mean())
        r["edge_disp_m"] = self.boundary_displacement(t["serv"], h["serv"],
                                                      None if mask is None else m)
        bt, bh = self._boundary(t["serv"]), self._boundary(h["serv"])
        if mask is not None:
            M = self._mask_raster(m)
            bt, bh = bt & M, bh & M
        r["handover_err"] = float(bh.sum() / max(bt.sum(), 1) - 1)
        return r

    def plan(self, PLh, mask=None):
        """Greedy site selection on the predicted coverage at PLAN_LEVEL, scored
        on the true coverage over the pixels in `mask`."""
        m = np.ones(len(self.true_pl), bool) if mask is None else np.asarray(mask, bool)
        cov_h = (PLh[m] <= PLAN_LEVEL).T                   # (sites, pixels)
        cov_t = (self.true_pl[m] <= PLAN_LEVEL).T
        out = []
        for K in PLAN_K:
            sel_h, _ = lazy_greedy_max_coverage(cov_h, K)
            sel_t, _ = lazy_greedy_max_coverage(cov_t, K)
            got = cov_t[sel_h].any(0).mean() if sel_h else 0.0
            best = cov_t[sel_t].any(0).mean() if sel_t else 0.0
            out.append(dict(K=K, cov_true_of_pred_plan=float(got),
                            cov_true_of_true_plan=float(best), regret=float(best - got)))
        # sites needed for COV_TARGET coverage: greedy order on each map
        S = cov_h.shape[0]
        ord_h, _ = lazy_greedy_max_coverage(cov_h, S)
        ord_t, _ = lazy_greedy_max_coverage(cov_t, S)

        def need(order, cov):
            c = np.zeros(cov.shape[1], bool)
            for i, j in enumerate(order, 1):
                c |= cov[j]
                if c.mean() >= COV_TARGET:
                    return i
            return np.nan                               # target not reachable
        k_h, k_t = need(ord_h, cov_h), need(ord_t, cov_t)
        sel = ord_h[: int(k_h)] if np.isfinite(k_h) else ord_h
        out.append(dict(K=-1, sites_pred=k_h, sites_true=k_t,
                        cov_true_of_pred_plan=float(cov_t[sel].any(0).mean()) if sel else 0.0,
                        cov_true_of_true_plan=float("nan"), regret=float("nan")))
        return out


def unsurveyed_mask(city, surveyed_tiles) -> np.ndarray:
    """Pixels outside the surveyed tiles (the few-shot evaluation set)."""
    return ~np.isin(city.tile, np.asarray(surveyed_tiles))


def load_city(cfg, base, name: str) -> City:
    raw, proc = Path(base.paths.raw), Path(base.paths.processed)
    meas, _ = load_links(raw / cfg.dataset / name / "measurements.parquet", cfg)
    tiles = pd.read_parquet(proc / cfg.dataset / name / "tiles.parquet")
    grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()),
                      float(cfg.tile_size_m))
    meta = json.loads((raw / cfg.scenes_dir / name / "scene_meta.json").read_text())
    return City(meas, grid, tiles, float(cfg.d0_m),
                LocalFrame.from_center(meta["lat0"], meta["lon0"]))
