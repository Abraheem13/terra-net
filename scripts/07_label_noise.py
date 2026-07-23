#!/usr/bin/env python
"""Per-tile label-noise floor for (gamma, PL0).

The tile labels are themselves estimates, so RMSE against them overstates true
model error. This computes two bounds on sigma(gamma):

  LOWER  analytic SE of the ridge-shrunk fit, propagated from residual variance:
         Var(theta) = s^2 M^-1 (A'A) M^-1,  M = A'A + lam*I.
         Verified against Monte-Carlo to within 1%. Assumes iid residuals, so
         it under-states noise where residuals are spatially correlated.

  UPPER  BS-grouped split-half: refit the SAME ridge procedure on two disjoint
         halves of the base stations and compare. Splitting by BS (not at
         random) keeps correlated rays together, so this captures the
         correlation the analytic SE ignores. Half-data fits are noisier than
         full-data fits, hence an upper bound.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from shapely.geometry import box

from terranet.data.tiling import build_grid, tiles_to_gdf
from terranet.models.baselines.log_distance import fit_global
from terranet.utils.logging import get_logger

log = get_logger("label_noise")
LAM = 50.0          # must match n_prior in fit_tiles_local_ridge
D0 = 1.0


def ridge_fit(x, y, theta_g, lam=LAM):
    """Returns (theta, se, resid_rmse). theta = [gamma, pl0]."""
    A = np.c_[x, np.ones(len(x))]
    M = A.T @ A + lam * np.eye(2)
    theta = np.linalg.solve(M, A.T @ y + lam * theta_g)
    resid = y - A @ theta
    dof = max(len(x) - 2, 1)
    s2 = float((resid ** 2).sum() / dof)
    Minv = np.linalg.inv(M)
    cov = s2 * (Minv @ (A.T @ A) @ Minv)
    return theta, np.sqrt(np.clip(np.diag(cov), 0, None)), float(np.sqrt((resid ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--tile-size", type=float, default=100.0)
    ap.add_argument("--min-half", type=int, default=30)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load("configs/base.yaml")
    min_n = int(cfg.min_measurements_per_tile)

    summary = []
    for city in cfg.cities:
        m = pd.read_parquet(Path(base.paths.raw) / cfg.dataset / city / "measurements.parquet")
        pad = 0.002
        region = box(m.rx_lon.min() - pad, m.rx_lat.min() - pad,
                     m.rx_lon.max() + pad, m.rx_lat.max() + pad)
        tgdf = tiles_to_gdf(build_grid(region, args.tile_size))
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(m.rx_lon, m.rx_lat), crs="EPSG:4326")
        idx = gpd.sjoin(pts, tgdf.reset_index(drop=True), how="left",
                        predicate="within")["index_right"].to_numpy()

        d = m.dist_m.to_numpy(float)
        pl = m.pathloss_db.to_numpy(float)
        x_all = 10.0 * np.log10(np.maximum(d, D0) / D0)
        bs = m.bs_id.to_numpy()

        g_glob, p_glob = fit_global(d, pl, d0=D0)
        theta_g = np.array([g_glob, p_glob])

        bs_names = np.array(sorted(pd.unique(bs)))
        half_a = set(bs_names[: len(bs_names) // 2])
        in_a = np.array([b in half_a for b in bs])

        n_t = len(tgdf)
        g_se = np.full(n_t, np.nan)
        p_se = np.full(n_t, np.nan)
        diffs_g, diffs_p = [], []

        idx_i = np.where(~pd.isna(idx), idx, -1).astype(int)
        for t in range(n_t):
            sel = np.flatnonzero(idx_i == t)
            if len(sel) < min_n:
                continue
            _, se, _ = ridge_fit(x_all[sel], pl[sel], theta_g)
            g_se[t], p_se[t] = se[0], se[1]

            a = sel[in_a[sel]]
            b = sel[~in_a[sel]]
            if len(a) >= args.min_half and len(b) >= args.min_half:
                ta, _, _ = ridge_fit(x_all[a], pl[a], theta_g)
                tb, _, _ = ridge_fit(x_all[b], pl[b], theta_g)
                diffs_g.append(ta[0] - tb[0])
                diffs_p.append(ta[1] - tb[1])

        tf = Path(base.paths.processed) / cfg.dataset / city / "tiles.parquet"
        td = pd.read_parquet(tf)
        if len(td) == n_t:
            td["gamma_se"] = g_se
            td["pl0_se"] = p_se
            td.to_parquet(tf)
        else:
            log.warning(f"{city}: tile count mismatch ({len(td)} vs {n_t}), SEs not merged")

        fitted = np.isfinite(g_se)
        analytic_g = float(np.sqrt(np.nanmean(g_se[fitted] ** 2)))
        analytic_p = float(np.sqrt(np.nanmean(p_se[fitted] ** 2)))
        split_g = float(np.std(diffs_g) / np.sqrt(2)) if diffs_g else np.nan
        split_p = float(np.std(diffs_p) / np.sqrt(2)) if diffs_p else np.nan

        summary.append(dict(city=city, n_tiles=int(fitted.sum()), n_split=len(diffs_g),
                            gamma_se_analytic=round(analytic_g, 4),
                            gamma_se_split=round(split_g, 4),
                            pl0_se_analytic=round(analytic_p, 3),
                            pl0_se_split=round(split_p, 3)))
        log.info(f"{city:11s} tiles={int(fitted.sum()):5d}  "
                 f"sigma(gamma): analytic={analytic_g:.3f}  bs-split={split_g:.3f}")

    df = pd.DataFrame(summary)
    out = Path(base.paths.outputs) / "tables" / "label_noise.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== label-noise floor per city ===")
    print(df.to_string(index=False))
    print(f"\ncorpus RMS sigma(gamma): analytic={np.sqrt((df.gamma_se_analytic**2).mean()):.3f}"
          f"  bs-split={np.sqrt((df.gamma_se_split**2).mean()):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
