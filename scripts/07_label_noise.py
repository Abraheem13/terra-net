#!/usr/bin/env python
"""Precision of the tile labels (gamma, PL0): three estimators per city.

  analytic   iid-residual standard error of the shrunk estimator
  cluster    cluster-robust (CR2) standard error, clusters = base stations
  split-half the base stations are split at random into two disjoint halves
             and each half is refitted INDEPENDENTLY (its own empirical-Bayes
             prior). sd(diff)/sqrt(2) is the standard error of a half-data
             fit; dividing by a further sqrt(2) extrapolates to the full data
             under the assumption that variance scales with 1/#stations.
             Repeated for R random partitions.

Analytic and cluster SEs come from tiles.parquet (02_build_tiles.py); the split
half is recomputed from the raw links with the identical grid and estimator.

Output: outputs/tables/label_noise.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.data.labels import fit_tiles, link_distance_m, load_links, log_distance
from terranet.data.tiling import grid_frame, point_tile_index, region_from_points
from terranet.experiments.common import load_cfg
from terranet.utils.logging import get_logger

log = get_logger("label_noise")
R = 5


def rms(a):
    a = a[np.isfinite(a)]
    return float(np.sqrt((a ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--min-half", type=int, default=30)
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    rows = []
    for ci, city in enumerate(cfg.cities):
        raw = Path(base.paths.raw) / cfg.dataset / city
        meas, _ = load_links(raw / "measurements.parquet", cfg)
        tiles = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / city / "tiles.parquet")
        grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()),
                          float(cfg.tile_size_m))
        assert len(grid) == len(tiles), "grid mismatch with tiles.parquet"
        idx = point_tile_index(grid, meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy())
        x = log_distance(link_distance_m(meas), float(cfg.d0_m))
        y = meas.pathloss_db.to_numpy(float)
        bs = meas.bs_id.to_numpy()
        names = np.array(sorted(pd.unique(bs)))
        dg, dp = [], []
        for r in range(R):
            rng = np.random.default_rng([ci, r])
            half = set(rng.permutation(names)[: len(names) // 2])
            a = np.array([b in half for b in bs])
            fa = fit_tiles(np.where(a, idx, -1), x, y, bs, len(grid), min_count=args.min_half)
            fb = fit_tiles(np.where(~a, idx, -1), x, y, bs, len(grid), min_count=args.min_half)
            ok = np.isfinite(fa.gamma) & np.isfinite(fb.gamma) & np.isfinite(tiles.gamma.to_numpy())
            dg.append(fa.gamma[ok] - fb.gamma[ok])
            dp.append(fa.pl0[ok] - fb.pl0[ok])
        dg, dp = np.concatenate(dg), np.concatenate(dp)
        half_g, half_p = float(np.std(dg) / np.sqrt(2)), float(np.std(dp) / np.sqrt(2))
        rows.append(dict(
            city=city, n_tiles=int(np.isfinite(tiles.gamma).sum()),
            n_split_pairs=int(len(dg) / R),
            gamma_se_analytic=rms(tiles.gamma_se.to_numpy()),
            gamma_se_cluster=rms(tiles.gamma_se_cl.to_numpy()),
            gamma_se_half=half_g, gamma_se_split_full=half_g / np.sqrt(2),
            pl0_se_analytic=rms(tiles.pl0_se.to_numpy()),
            pl0_se_cluster=rms(tiles.pl0_se_cl.to_numpy()),
            pl0_se_half=half_p, pl0_se_split_full=half_p / np.sqrt(2),
            median_n_bs=float(np.nanmedian(tiles.n_bs.where(np.isfinite(tiles.gamma))))))
        r = rows[-1]
        log.info(f"{city:10s} sigma(gamma): analytic={r['gamma_se_analytic']:.3f} "
                 f"cluster={r['gamma_se_cluster']:.3f} split(full)={r['gamma_se_split_full']:.3f}")
    out = Path(base.paths.outputs) / "tables" / "label_noise.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
