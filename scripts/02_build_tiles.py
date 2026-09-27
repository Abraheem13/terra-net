#!/usr/bin/env python
"""Tile every city, fit per-tile (gamma, PL0) labels and compute descriptors.

Inputs  (per city)
  data/raw/sionna/<city>/measurements.parquet        ray-traced links (01b)
  data/raw/sionna_scenes/<city>/building.geojson     scene geometry (01a)
  data/raw/sionna_scenes/<city>/scene_meta.json
Outputs
  data/processed/sionna/<city>/tiles.parquet          one row per tile
  outputs/tables/corpus.csv                            per-city corpus summary
  outputs/tables/fits.csv                              per-city estimation summary

Labels: empirical-Bayes shrinkage (terranet.data.labels), 3-D distance,
reference distance d0 from the config. Descriptors: 32 raw built-environment
features from the exact scene geometry (terranet.data.descriptors.scene_tiles).
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import scene_building_parts, tile_descriptors
from terranet.data.labels import fit_global, fit_tiles, link_distance_m, load_links, log_distance
from terranet.data.sionna_gen.geo import LocalFrame
from terranet.data.sionna_gen.osm_scene import resolve_heights
from terranet.data.tiling import grid_frame, point_tile_index, region_from_points
from terranet.experiments.common import load_cfg
from terranet.utils.logging import get_logger

log = get_logger("tiles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--city", default=None)
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    raw, proc = Path(base.paths.raw), Path(base.paths.processed)
    out_tab = Path(base.paths.outputs) / "tables"
    out_tab.mkdir(parents=True, exist_ok=True)
    tile_m, d0, n_min = float(cfg.tile_size_m), float(cfg.d0_m), int(cfg.min_measurements_per_tile)

    corpus, fits = [], []
    for city in cfg.cities:
        if args.city and city != args.city:
            continue
        meas, n_qc = load_links(raw / cfg.dataset / city / "measurements.parquet", cfg)
        scene_dir = raw / cfg.scenes_dir / city
        meta = json.loads((scene_dir / "scene_meta.json").read_text())

        grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()),
                          tile_m)
        idx = point_tile_index(grid, meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy())
        x = log_distance(link_distance_m(meas), d0)
        y = meas.pathloss_db.to_numpy(float)
        f = fit_tiles(idx, x, y, meas.bs_id.to_numpy(), len(grid), min_count=n_min)
        g_glob, p_glob = fit_global(x, y)

        frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
        bld = gpd.read_file(scene_dir / "building.geojson")
        bld = bld[bld.geometry.notna()].reset_index(drop=True)
        parts, heights = scene_building_parts(bld, frame, float(meta["size_m"]))
        D = tile_descriptors(grid, parts, heights, frame)

        tiles = pd.concat([grid.reset_index(drop=True), f.frame(), D], axis=1)
        tiles.insert(0, "city", city)
        od = proc / cfg.dataset / city
        od.mkdir(parents=True, exist_ok=True)
        tiles.to_parquet(od / "tiles.parquet")

        _, prov = resolve_heights(bld)
        ok = np.isfinite(f.gamma)
        wrmse = float(np.sqrt(np.nansum(f.fit_rmse_db[ok] ** 2 * f.n[ok]) / f.n[ok].sum()))
        corpus.append(dict(
            city=city, buildings=prov["n_buildings"],
            frac_tag=prov["frac_height_tag"], frac_levels=prov["frac_levels"],
            frac_imputed=prov["frac_imputed"], median_height_m=prov["median_height_m"],
            building_parts=len(parts), transmitters=int(meas.bs_id.nunique()),
            samples=int(len(meas)), qc_removed=n_qc, tiles=int(ok.sum()),
            tiles_grid=len(grid)))
        fits.append(dict(
            city=city, gamma_global=g_glob, pl0_global=p_glob,
            mu=f.mu, tau=float(np.sqrt(f.tau2)),
            gamma_median=float(np.nanmedian(f.gamma)), gamma_sd=float(np.nanstd(f.gamma)),
            pl0_median=float(np.nanmedian(f.pl0)),
            fit_rmse_db=wrmse, shrink_median=float(np.nanmedian(f.shrink)),
            gamma_ols_min=float(np.nanmin(f.gamma_ols)),
            gamma_ols_max=float(np.nanmax(f.gamma_ols)),
            gamma_min=float(np.nanmin(f.gamma)), gamma_max=float(np.nanmax(f.gamma)),
            median_n_bs=float(np.nanmedian(np.where(ok, f.n_bs, np.nan)))))
        log.info(f"{city:10s} samples={len(meas):>9,d} tiles={ok.sum():5d} "
                 f"gamma med={fits[-1]['gamma_median']:.2f} tau={fits[-1]['tau']:.2f} "
                 f"fitRMSE={wrmse:.2f} dB")

    if not args.city:
        pd.DataFrame(corpus).to_csv(out_tab / "corpus.csv", index=False)
        pd.DataFrame(fits).to_csv(out_tab / "fits.csv", index=False)
        log.info(f"wrote {out_tab/'corpus.csv'} and {out_tab/'fits.csv'}")


if __name__ == "__main__":
    main()
