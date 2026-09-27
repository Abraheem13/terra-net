"""Per-tile built-environment descriptors computed from the ray-traced scene geometry.

The descriptor must describe exactly the geometry the propagation solver saw,
so this module applies the same rules as `sionna_gen.osm_scene.build_scene`:

  * heights from `resolve_heights` (OSM height tag, else 3.2 m per storey,
    else the per-city median) and clipped to [2, 500] m;
  * footprint exteriors only (courtyards are filled in the extruded mesh);
  * geometry clipped to the scene square and parts under 4 m^2 dropped;
  * each footprint part clipped to the tile it falls in (no double counting
    of area across tiles).

Only the 32 built-environment features are produced. The scenes contain no
terrain relief, water or vegetation surfaces and no weather, so topographic,
land-cover and climatic descriptors would describe nothing the solver
modelled; they are deliberately excluded.

Values are RAW (unnormalised). Normalisation is fitted on source cities
inside each experiment, never per city.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, box

from terranet.data.sionna_gen.geo import LocalFrame
from terranet.data.sionna_gen.osm_scene import _polys, resolve_heights

from .built_env import BUILT_NAMES, built_env_features

DESCRIPTOR_NAMES = list(BUILT_NAMES)
DESCRIPTOR_COLUMNS = [f"d_{n}" for n in DESCRIPTOR_NAMES]
MIN_PART_M2 = 4.0


def scene_building_parts(buildings: GeoDataFrame, frame: LocalFrame, size_m: float):
    """Footprint parts in local metres, exactly as meshed by the scene builder."""
    heights, _ = resolve_heights(buildings.reset_index(drop=True))
    clip = box(-size_m / 2, -size_m / 2, size_m / 2, size_m / 2)
    geoms, hs = [], []
    for geom, h in zip(buildings.geometry, heights, strict=False):
        for poly in _polys(geom):
            x, y = frame.to_local(*np.asarray(poly.exterior.coords).T)
            p = Polygon(np.c_[x, y]).buffer(0).intersection(clip)
            for q in _polys(p):
                if q.area >= MIN_PART_M2:
                    geoms.append(q)
                    hs.append(float(h))
    return np.asarray(geoms, dtype=object), np.asarray(hs, float)


def tile_polygons_local(grid: pd.DataFrame, frame: LocalFrame) -> np.ndarray:
    polys = []
    for r in grid.itertuples():
        lon = np.array([r.minx, r.maxx, r.maxx, r.minx])
        lat = np.array([r.miny, r.miny, r.maxy, r.maxy])
        x, y = frame.to_local(lon, lat)
        polys.append(Polygon(np.c_[x, y]))
    return np.asarray(polys, dtype=object)


def tile_descriptors(grid: pd.DataFrame, parts: np.ndarray, heights: np.ndarray,
                     frame: LocalFrame) -> pd.DataFrame:
    """(n_tiles, 32) raw built-environment descriptors."""
    tiles = tile_polygons_local(grid, frame)
    tree = shapely.STRtree(parts) if len(parts) else None
    out = np.zeros((len(tiles), len(DESCRIPTOR_NAMES)), np.float32)
    for i, tp in enumerate(tiles):
        if tree is None:
            continue
        hit = tree.query(tp, predicate="intersects")
        if hit.size == 0:
            continue
        clipped = shapely.intersection(parts[hit], tp)
        area = shapely.area(clipped)
        keep = area >= 1e-6
        if not keep.any():
            continue
        gdf = GeoDataFrame({"height": heights[hit][keep], "area_m2": area[keep]},
                           geometry=list(clipped[keep]))
        out[i] = built_env_features(gdf, float(tp.area))
    return pd.DataFrame(out, columns=DESCRIPTOR_COLUMNS)
