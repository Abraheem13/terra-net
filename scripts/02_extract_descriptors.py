#!/usr/bin/env python
"""Phase 1: tile each city, fit per-tile (gamma, PL0), write tiles.parquet.

--fit local  (default): per-tile fit on RX points inside the tile. Well-posed
             even with a single BS.
--fit mixed: joint mixed-path LS across LoS segments (needs multiple BSs to be
             identifiable; use for city scenarios).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from shapely.geometry import LineString, box

from terranet.data.tiling import build_grid, tiles_to_gdf
from terranet.models.baselines.log_distance import fit_tiles_local_ridge, fit_tiles_mixed_path
from terranet.utils.geo import haversine_m
from terranet.utils.logging import get_logger

log = get_logger("extract")


def rx_tile_index(tgdf, meas):
    import geopandas as gpd
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(meas.rx_lon, meas.rx_lat), crs="EPSG:4326")
    joined = gpd.sjoin(pts, tgdf.reset_index(drop=True), how="left", predicate="within")
    return joined["index_right"].to_numpy()


def segment_measurements(tgdf, meas):
    sindex = tgdf.sindex
    segments, pl = [], []
    for row in meas.itertuples():
        line = LineString([(row.tx_lon, row.tx_lat), (row.rx_lon, row.rx_lat)])
        segs = []
        for h in sindex.query(line, predicate="intersects"):
            inter = line.intersection(tgdf.geometry.iloc[h])
            if not inter.is_empty and inter.length > 0:
                segs.append((int(h), inter.length * 111_000.0))
        if segs:
            segments.append(segs)
            pl.append(row.pathloss_db)
    return segments, np.asarray(pl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/deepmimo.yaml")
    ap.add_argument("--tile-size", type=float, default=200.0)
    ap.add_argument("--fit", choices=["local", "mixed"], default="local")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load("configs/base.yaml")

    for city in cfg.cities:
        meas_f = Path(base.paths.raw) / cfg.dataset / city / "measurements.parquet"
        if not meas_f.exists():
            log.warning(f"skip {city}: {meas_f} missing")
            continue
        meas = pd.read_parquet(meas_f)
        pad = 0.002
        region = box(meas.rx_lon.min() - pad, meas.rx_lat.min() - pad,
                     meas.rx_lon.max() + pad, meas.rx_lat.max() + pad)
        tiles = build_grid(region, args.tile_size)
        tgdf = tiles_to_gdf(tiles)

        if args.fit == "local":
            idx = rx_tile_index(tgdf, meas)
            ok = ~pd.isna(idx)
            d = haversine_m(np.radians(meas.tx_lat), np.radians(meas.tx_lon),
                            np.radians(meas.rx_lat), np.radians(meas.rx_lon)).to_numpy()
            gamma, pl0, counts, rmse, shrink, theta_g = fit_tiles_local_ridge(
                idx[ok].astype(int), d[ok], meas.pathloss_db.to_numpy()[ok],
                len(tgdf), min_count=int(cfg.min_measurements_per_tile))
            log.info(f"{city}: global fit gamma={theta_g[0]:.2f} pl0={theta_g[1]:.1f}")
        else:
            segments, pl = segment_measurements(tgdf, meas)
            gamma, pl0, counts = fit_tiles_mixed_path(segments, pl, len(tgdf))
            rmse = np.full(len(tgdf), np.nan)

        df = tgdf.drop(columns="geometry").copy()
        df["gamma"], df["pl0"] = gamma, pl0
        df["n_measurements"], df["fit_rmse_db"] = counts, rmse
        df["shrinkage"] = shrink if args.fit == "local" else np.nan
        df["raster_idx"] = np.arange(len(df))
        desc = pd.DataFrame(np.zeros((len(df), 116), np.float32),
                            columns=[f"descriptor_{i}" for i in range(116)])
        df = pd.concat([df.reset_index(drop=True), desc], axis=1)
        out = Path(base.paths.processed) / cfg.dataset / city
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "tiles.parquet")
        fitted = np.isfinite(gamma)
        log.info(f"{city}: {fitted.sum()}/{len(df)} tiles fitted "
                 f"(median gamma={np.nanmedian(gamma):.2f}, "
                 f"median fit RMSE={np.nanmedian(rmse):.2f} dB)")


if __name__ == "__main__":
    main()
