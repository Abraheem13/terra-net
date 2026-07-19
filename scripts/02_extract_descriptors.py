#!/usr/bin/env python
"""Phase 1: tile each city, fit per-tile (gamma, PL0) via mixed-path LS,
extract 116-dim descriptors + rasters, write tiles.parquet + rasters.zarr.

Expects normalized measurement parquet per city in data/raw/<dataset>/<city>/measurements.parquet
with columns: tx_lat, tx_lon, rx_lat, rx_lon, pathloss_db  (degrees).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from shapely.geometry import LineString, box

from terranet.data.tiling import build_grid, assign_structures, tiles_to_gdf
from terranet.models.baselines.log_distance import fit_tiles_mixed_path
from terranet.utils.logging import get_logger

log = get_logger("extract")


def segment_measurements(tiles_gdf, meas: pd.DataFrame):
    """Split each Tx-Rx LoS trace across intersected tiles -> segment lists."""
    sindex = tiles_gdf.sindex
    keys = list(zip(tiles_gdf["m"], tiles_gdf["z"]))
    key_to_idx = {k: i for i, k in enumerate(keys)}
    segments, pl = [], []
    for row in meas.itertuples():
        line = LineString([(row.tx_lon, row.tx_lat), (row.rx_lon, row.rx_lat)])
        hits = sindex.query(line, predicate="intersects")
        segs = []
        for h in hits:
            inter = line.intersection(tiles_gdf.geometry.iloc[h])
            if inter.is_empty:
                continue
            # approx metres: 1 deg ~ 111 km (fine within a city; refine with pyproj if needed)
            seg_len = inter.length * 111_000.0
            if seg_len > 0:
                segs.append((h, seg_len))
        if segs:
            segments.append(segs)
            pl.append(row.pathloss_db)
    return segments, np.asarray(pl), key_to_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/deepmimo.yaml")
    ap.add_argument("--tile-size", type=float, default=200.0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load("configs/base.yaml")

    for city in cfg.cities:
        raw = Path(base.paths.raw) / cfg.dataset / city
        meas_f = raw / "measurements.parquet"
        if not meas_f.exists():
            log.warning(f"skip {city}: {meas_f} missing")
            continue
        meas = pd.read_parquet(meas_f)
        pad = 0.01
        region = box(meas.rx_lon.min() - pad, meas.rx_lat.min() - pad,
                     meas.rx_lon.max() + pad, meas.rx_lat.max() + pad)
        tiles = build_grid(region, args.tile_size)
        tgdf = tiles_to_gdf(tiles)
        segments, pl, _ = segment_measurements(tgdf, meas)
        gamma, pl0, counts = fit_tiles_mixed_path(segments, pl, len(tgdf))
        out = Path(base.paths.processed) / cfg.dataset / city
        out.mkdir(parents=True, exist_ok=True)
        df = tgdf.drop(columns="geometry").copy()
        df["gamma"], df["pl0"], df["n_measurements"] = gamma, pl0, counts
        df["raster_idx"] = np.arange(len(df))
        # Descriptors + rasters require structure/DEM inputs; see docs/DESCRIPTORS.md.
        for i in range(116):
            df[f"descriptor_{i}"] = 0.0   # filled by the descriptor pass
        df.to_parquet(out / "tiles.parquet")
        log.info(f"{city}: {len(df)} tiles, {int(counts.sum())} segment contributions")


if __name__ == "__main__":
    main()
