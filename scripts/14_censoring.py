#!/usr/bin/env python
"""Effect of the 150 dB path-loss cut-off on the tile labels.

The ray tracer reports links up to the cut-off only, so far or deeply
shadowed links of a tile are missing, which biases a least-squares slope
downwards. Three measurements per city:
  censored   share of (pixel, site) pairs of the labelled tiles beyond the cut
  cutoff     tile labels refitted with the cut lowered to 140 and 130 dB;
             median change of gamma against the 150 dB labels (same tiles)
  tobit      per tile (random subset), the slope of a censored-Gaussian
             (Tobit) log-distance model that uses the censored pairs, against
             the least-squares slope on the observed pairs only (both
             unshrunk, same x = 10 log10(d / d0))

Output: outputs/tables/censoring.csv (one row per city)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr

from terranet.data.labels import fit_tiles, link_distance_m, load_links, log_distance
from terranet.data.tiling import grid_frame, point_tile_index, region_from_points
from terranet.evaluation.network import load_city
from terranet.experiments.common import load_cfg
from terranet.utils.logging import get_logger

log = get_logger("censoring")
CUTS = (140.0, 130.0)
N_TOBIT = 300


def tobit_slope(x, y, cens, c):
    """ML slope of y = a + g x + s e, e ~ N(0, 1), right-censored at c."""
    xo, yo, xc = x[~cens], y[~cens], x[cens]
    A = np.c_[np.ones_like(xo), xo]
    (a0, g0), *_ = np.linalg.lstsq(A, yo, rcond=None)
    s0 = max(np.std(yo - a0 - g0 * xo), 1.0)

    def nll(p):
        a, g, ls = p
        s = np.exp(ls)
        zo = (yo - a - g * xo) / s
        zc = (c - a - g * xc) / s
        return 0.5 * (zo ** 2).sum() + len(yo) * ls - log_ndtr(-zc).sum()

    r = minimize(nll, [a0, g0, np.log(s0)], method="L-BFGS-B")
    return float(g0), float(r.x[1]), bool(r.success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    cut0 = 150.0
    rows = []
    for ci, name in enumerate(cfg.cities):
        meas, _ = load_links(Path(base.paths.raw) / cfg.dataset / name / "measurements.parquet",
                             cfg)
        grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()),
                          float(cfg.tile_size_m))
        idx = point_tile_index(grid, meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy())
        x = log_distance(link_distance_m(meas), float(cfg.d0_m))
        y = meas.pathloss_db.to_numpy(float)
        bs = meas.bs_id.to_numpy()
        n_min = int(cfg.min_measurements_per_tile)
        f0 = fit_tiles(idx, x, y, bs, len(grid), min_count=n_min)
        r = dict(city=name, tiles=int(np.isfinite(f0.gamma).sum()),
                 gamma_median_150=float(np.nanmedian(f0.gamma)))
        for c in CUTS:
            keep = y <= c
            f = fit_tiles(np.where(keep, idx, -1), x, y, bs, len(grid), min_count=n_min)
            both = np.isfinite(f.gamma) & np.isfinite(f0.gamma)
            r[f"tiles_{int(c)}"] = int(both.sum())
            r[f"dgamma_median_{int(c)}"] = float(np.median(f.gamma[both] - f0.gamma[both]))
            r[f"links_removed_{int(c)}"] = float(1 - keep.mean())

        city = load_city(cfg, base, name)
        cens = ~np.isfinite(city.true_pl)
        r["censored_share"] = float(cens.mean())
        rng = np.random.default_rng(ci)
        tiles = np.unique(city.tile)
        g_ls, g_tb = [], []
        for t in rng.choice(tiles, size=min(N_TOBIT, len(tiles)), replace=False):
            m = city.tile == t
            xt, yt, ct = city.x[m].ravel(), city.true_pl[m].ravel(), cens[m].ravel()
            if (~ct).sum() < n_min:
                continue
            gl, gt, ok = tobit_slope(xt, np.where(ct, cut0, yt), ct, cut0)
            if ok:
                g_ls.append(gl)
                g_tb.append(gt)
        g_ls, g_tb = np.array(g_ls), np.array(g_tb)
        r.update(tobit_tiles=len(g_ls), gamma_ls_median=float(np.median(g_ls)),
                 gamma_tobit_median=float(np.median(g_tb)),
                 dgamma_tobit_median=float(np.median(g_tb - g_ls)),
                 dgamma_tobit_q90=float(np.quantile(g_tb - g_ls, 0.9)))
        rows.append(r)
        log.info(f"{name:10s} censored={r['censored_share']:.3f} "
                 f"dgamma(140)={r['dgamma_median_140']:+.3f} "
                 f"dgamma(tobit)={r['dgamma_tobit_median']:+.3f}")
    out = Path(base.paths.outputs) / "tables" / "censoring.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
