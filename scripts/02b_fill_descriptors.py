#!/usr/bin/env python
"""Fill built-environment + land-cover descriptors into tiles.parquet.

Sources OSM either live (Overpass, cached) or from local GeoJSON files in
data/raw/osm/<city>/ when --local is passed (use when the cluster network is
unreliable: fetch on a laptop, scp up). Topographic + climate dims stay zero
for now — documented limitation, they matter cross-city not within-city.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from shapely.geometry import box

from terranet.data.descriptors.built_env import built_env_features, landcover_features
from terranet.data.tiling import build_grid, tiles_to_gdf
from terranet.utils.logging import get_logger

log = get_logger("descriptors")

TAGSETS = [
    ({"building": True}, "building"),
    ({"natural": "water"}, "water"),
    ({"landuse": ["grass", "forest", "meadow"], "leisure": "park", "natural": "wood"},
     "vegetation"),
]


def fetch_osm_live(bbox, cache_dir: Path):
    import osmnx as ox
    ox.settings.requests_timeout = 300
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir / "osmnx_cache")
    west, south, east, north = bbox
    frames = []
    for tags, kind in TAGSETS:
        try:
            g = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
            g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            g["kind"] = kind
            keep = ["geometry", "kind"] + [c for c in ("height", "building:levels")
                                           if c in g.columns]
            g = g[keep]
            g.to_file(cache_dir / f"{kind}.geojson", driver="GeoJSON")
            frames.append(g)
            log.info(f"  fetched {kind}: {len(g)}")
        except Exception as e:
            log.warning(f"  {kind}: {type(e).__name__}: {e}")
    return frames


def load_osm_local(cache_dir: Path):
    frames = []
    for _, kind in TAGSETS:
        f = cache_dir / f"{kind}.geojson"
        if f.exists():
            g = gpd.read_file(f)
            g["kind"] = kind
            frames.append(g)
            log.info(f"  loaded {kind}: {len(g)}")
        else:
            log.warning(f"  missing {f}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tile-size", type=float, required=True)
    ap.add_argument("--local", action="store_true", help="read GeoJSON instead of fetching")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load("configs/base.yaml")

    for city in cfg.cities:
        tiles_f = Path(base.paths.processed) / cfg.dataset / city / "tiles.parquet"
        meas = pd.read_parquet(Path(base.paths.raw) / cfg.dataset / city / "measurements.parquet")
        pad = 0.002
        bbox = (float(meas.rx_lon.min() - pad), float(meas.rx_lat.min() - pad),
                float(meas.rx_lon.max() + pad), float(meas.rx_lat.max() + pad))
        log.info(f"{city}: bbox={tuple(round(b, 4) for b in bbox)}")
        cache_dir = (Path(base.paths.raw) / "sionna_scenes" / city
             if cfg.dataset == "sionna"
             else Path(base.paths.raw) / "osm" / city)
        cache_dir.mkdir(parents=True, exist_ok=True)

        frames = load_osm_local(cache_dir) if args.local else fetch_osm_live(bbox, cache_dir)
        if not frames:
            log.error(f"{city}: no OSM features — skipping")
            continue
        osm = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")

        utm = osm.estimate_utm_crs()
        osm_m = osm.to_crs(utm)
        osm["area_m2"] = osm_m.area
        h = pd.to_numeric(osm["height"], errors="coerce") if "height" in osm else pd.Series(np.nan, index=osm.index)
        lv = pd.to_numeric(osm["building:levels"], errors="coerce") * 3.0 if "building:levels" in osm else pd.Series(np.nan, index=osm.index)
        osm["height"] = h.fillna(lv).fillna(9.0)

        tiles = build_grid(box(*bbox), args.tile_size)
        tgdf = tiles_to_gdf(tiles)
        area = args.tile_size ** 2
        sindex = osm.sindex
        D = np.zeros((len(tgdf), 116), np.float32)
        for i, geom in enumerate(tgdf.geometry):
            hits = list(sindex.query(geom, predicate="intersects"))
            if not hits:
                continue
            sub = osm.iloc[hits]
            b = sub[sub.kind == "building"]
            if len(b):
                b_m = gpd.GeoDataFrame(
                    {"height": b.height.values, "area_m2": b.area_m2.values},
                    geometry=osm_m.geometry.iloc[b.index].values, crs=utm)
                D[i, 6:38] = built_env_features(b_m, area)
            D[i, 38:44] = landcover_features(sub[sub.kind == "water"],
                                             sub[sub.kind == "vegetation"], area)
        mx = np.abs(D).max(0)
        D = D / np.where(mx > 0, mx, 1.0)

        df = pd.read_parquet(tiles_f)
        df = df.drop(columns=[c for c in df.columns if c.startswith("descriptor_")])
        desc = pd.DataFrame(D, columns=[f"descriptor_{j}" for j in range(116)])
        df = pd.concat([df.reset_index(drop=True), desc], axis=1)
        df.to_parquet(tiles_f)
        log.info(f"{city}: written, {(D.sum(0) != 0).sum()}/116 dims non-zero")


if __name__ == "__main__":
    main()
