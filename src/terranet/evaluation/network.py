"""Network-level evaluation of predicted path-loss maps against ray tracing.

A `City` holds, for one city, every pixel (ray-traced cell inside a labelled
tile) and every site, the true path-loss matrix and the pixel-site geometry.
Any predictor that produces a (pixel x site) path-loss matrix is scored with
`City.evaluate` (path loss, serving site, coverage, SINR, spectral
efficiency) and `City.plan` (greedy site selection scored on the truth).

Link budget (downlink, full buffer, all sites active): P_tx = 46 dBm,
20 MHz, noise figure 7 dB -> N = -94.0 dBm. Pairs absent from the ray-traced
data exceeded the 150 dB cut-off (received power 10 dB below the noise floor)
and are treated as zero received power. Spectral efficiency: 3GPP TR 36.942
attenuated Shannon, 0.6 log2(1 + SINR), zero below -10 dB, capped at 4.4 b/s/Hz.
SINR below -20 dB, including pixels where no site is predicted to be
received, is set to -20 dB (outage) for the truth and every predictor alike.
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
COV_LEVELS = (120.0, 130.0, 140.0)
PLAN_LEVEL = 140.0
PLAN_K = (3, 5, 8)
COV_TARGET = 0.90          # sites needed to cover this fraction at PLAN_LEVEL
SINR_FLOOR_DB = -20.0      # outage: lower SINR (or no received site) counts as this value


def se_tr36942(sinr_db):
    s = 0.6 * np.log2(1 + 10 ** (sinr_db / 10))
    return np.where(sinr_db < -10, 0.0, np.minimum(s, 4.4))


class City:
    """Pixel x site geometry and the ray-traced truth for one city."""

    def __init__(self, meas, grid, tiles, d0, frame):
        pix, self.pix_id = np.unique(np.c_[meas.rx_lat.to_numpy(), meas.rx_lon.to_numpy()],
                                     axis=0, return_inverse=True)
        self.pix_id = self.pix_id.ravel()
        sites, self.site_id = np.unique(meas.bs_id.to_numpy(), return_inverse=True)
        self.site_id = self.site_id.ravel()
        P, S = len(pix), len(sites)
        tile = point_tile_index(grid, pix[:, 1], pix[:, 0])
        fitted = np.zeros(len(grid), bool)
        fitted[tiles.index[np.isfinite(tiles.gamma)]] = True
        keep = (tile >= 0) & fitted[np.maximum(tile, 0)]
        remap = np.full(P, -1)
        remap[keep] = np.arange(keep.sum())
        self.tile = tile[keep]
        # true path loss (inf where the pair was cut off)
        T = np.full((keep.sum(), S), np.inf)
        li = remap[self.pix_id]
        lk = li >= 0
        T[li[lk], self.site_id[lk]] = meas.pathloss_db.to_numpy(float)[lk]
        self.true_pl = T
        # distances pixel -> site in the scene's own metric frame (as the solver)
        tx = meas.groupby(self.site_id)[["tx_lat", "tx_lon", "tx_h"]].first().to_numpy()
        rx_h = float(meas.rx_h.iloc[0])
        px, py = frame.to_local(pix[keep, 1], pix[keep, 0])
        sx, sy = frame.to_local(tx[:, 1], tx[:, 0])
        self.px, self.py, self.rx_h = np.asarray(px), np.asarray(py), rx_h
        self.sx, self.sy, self.sz = np.asarray(sx), np.asarray(sy), tx[:, 2]
        self.sites = sites
        # raster indices of the pixels (cell centres on the solver grid)
        cell = float(np.median(np.diff(np.unique(np.round(self.px, 3)))))
        self.cell = cell
        self.ix = np.round((self.px - self.px.min()) / cell).astype(int)
        self.iy = np.round((self.py - self.py.min()) / cell).astype(int)
        dx = px[:, None] - sx[None, :]
        dy = py[:, None] - sy[None, :]
        dz = rx_h - tx[None, :, 2]
        self.x = log_distance(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), d0)
        # consistency with the solver distance on observed links
        d_obs = link_distance_m(meas)[lk]
        d_rec = 10 ** (self.x[li[lk], self.site_id[lk]] / 10) * d0
        self.dist_rel_err = float(np.max(np.abs(d_rec / d_obs - 1)))
        self.obs = np.isfinite(T)
        self.truth = self.summary(T)

    def summary(self, PL):
        rx = P_TX_DBM - PL                         # dBm, -inf where cut off
        lin = np.where(np.isfinite(rx), 10 ** (rx / 10), 0.0)
        serv = np.argmax(lin, 1)
        s = lin[np.arange(len(lin)), serv]
        sinr = 10 * np.log10(np.maximum(s, 1e-30) / (lin.sum(1) - s + 10 ** (NOISE_DBM / 10)))
        sinr = np.maximum(sinr, SINR_FLOOR_DB)
        return dict(serv=serv, serv_pl=PL[np.arange(len(PL)), serv], sinr=sinr,
                    se=se_tr36942(sinr))

    def subset(self, site_idx):
        """The same city served by a subset of the sites (pixels unchanged)."""
        c = copy.copy(self)
        site_idx = np.asarray(site_idx)
        c.true_pl, c.x = self.true_pl[:, site_idx], self.x[:, site_idx]
        c.sx, c.sy, c.sz = self.sx[site_idx], self.sy[site_idx], self.sz[site_idx]
        c.sites, c.obs = self.sites[site_idx], self.obs[:, site_idx]
        c.truth = c.summary(c.true_pl)
        return c

    def _boundary(self, serv):
        """Boolean raster of cell-edge pixels: the pixel's serving site differs
        from that of its east or north neighbour (one pixel per boundary
        crossing, so edges are one pixel thick)."""
        nx, ny = self.ix.max() + 1, self.iy.max() + 1
        A = np.full((nx, ny), -1)
        A[self.ix, self.iy] = serv
        B = np.zeros_like(A, bool)
        B[:-1] |= (A[1:] != A[:-1]) & (A[1:] >= 0) & (A[:-1] >= 0)
        B[:, :-1] |= (A[:, 1:] != A[:, :-1]) & (A[:, 1:] >= 0) & (A[:, :-1] >= 0)
        return B

    def boundary_displacement(self, serv_true, serv_pred):
        """Mean distance (m) between true and predicted cell edges, averaged
        over both directions (symmetric mean chamfer distance)."""
        bt, bp = self._boundary(serv_true), self._boundary(serv_pred)
        if not bt.any() or not bp.any():
            return float("nan")
        dt = distance_transform_edt(~bp, sampling=self.cell)[bt].mean()
        dp = distance_transform_edt(~bt, sampling=self.cell)[bp].mean()
        return float((dt + dp) / 2)

    def serving_residual(self, PLh):
        """Per pixel: predicted serving path loss, the residual r = true minus
        predicted serving path loss, and the true serving path loss. Coverage at
        level L holds exactly when predicted + r <= L."""
        h = self.summary(PLh)
        with np.errstate(invalid="ignore"):
            r = self.truth["serv_pl"] - h["serv_pl"]
        return h["serv_pl"], r, self.truth["serv_pl"]

    def predicted_pl(self, gamma_t, pl0_t):
        return pl0_t[self.tile][:, None] + gamma_t[self.tile][:, None] * self.x

    def evaluate(self, PLh, pl_for_error=None):
        """Network metrics of the predicted map PLh (inf = predicted beyond the
        cut-off). The path-loss error is computed over the ray-traced links from
        `pl_for_error` when given (the regression part of a hurdle model)."""
        t, h = self.truth, self.summary(PLh)
        E = (PLh if pl_for_error is None else pl_for_error) - self.true_pl
        r = {"pl_rmse_db": float(np.sqrt((E[self.obs] ** 2).mean())),
             "pl_bias_db": float(E[self.obs].mean()),
             "assoc_acc": float((h["serv"] == t["serv"]).mean())}
        for L in COV_LEVELS:
            ct, ch = t["serv_pl"] <= L, h["serv_pl"] <= L
            r[f"cov_true_{int(L)}"] = float(ct.mean())
            r[f"cov_acc_{int(L)}"] = float((ct == ch).mean())
            r[f"cov_err_{int(L)}"] = float(ch.mean() - ct.mean())
        r["sinr_mae_db"] = float(np.abs(h["sinr"] - t["sinr"]).mean())
        a, b = np.sort(t["sinr"]), np.sort(h["sinr"])
        grid = np.linspace(min(a[0], b[0]), max(a[-1], b[-1]), 2000)
        r["sinr_ks"] = float(np.abs(np.searchsorted(a, grid, "right") / len(a)
                                    - np.searchsorted(b, grid, "right") / len(b)).max())
        r["se_true"] = float(t["se"].mean())
        r["se_err"] = float(h["se"].mean() - t["se"].mean())
        r["edge_disp_m"] = self.boundary_displacement(t["serv"], h["serv"])
        r["handover_err"] = float(self._boundary(h["serv"]).sum()
                                  / max(self._boundary(t["serv"]).sum(), 1) - 1)
        return r

    def plan(self, PLh):
        cov_h = (PLh <= PLAN_LEVEL).T                   # (sites, pixels)
        cov_t = (self.true_pl <= PLAN_LEVEL).T
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




def load_city(cfg, base, name: str) -> City:
    raw, proc = Path(base.paths.raw), Path(base.paths.processed)
    meas, _ = load_links(raw / cfg.dataset / name / "measurements.parquet", cfg)
    tiles = pd.read_parquet(proc / cfg.dataset / name / "tiles.parquet")
    grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()),
                      float(cfg.tile_size_m))
    meta = json.loads((raw / cfg.scenes_dir / name / "scene_meta.json").read_text())
    return City(meas, grid, tiles, float(cfg.d0_m),
                LocalFrame.from_center(meta["lat0"], meta["lon0"]))
