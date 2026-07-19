"""Spherical grid construction over a convex polygon (base paper Algorithm 2).

Produces indexed tiles T_{m,z} with centres C_{m,z} = (phi0 + m*dphi, lambda0 + z*dlam),
EPSG:4326 / ISO 6709 ordering (m>0 north, z>0 east). Structures crossing tile
boundaries are split with shapely and reassigned per tile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

from terranet.utils.geo import angular_spans, spherical_centroid


@dataclass
class Tile:
    m: int
    z: int
    lat_c: float  # radians
    lon_c: float  # radians
    dphi: float
    dlam: float
    structures: gpd.GeoDataFrame | None = field(default=None, repr=False)

    @property
    def bounds_deg(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) in degrees."""
        lat, lon = np.degrees(self.lat_c), np.degrees(self.lon_c)
        hphi, hlam = np.degrees(self.dphi / 2), np.degrees(self.dlam / 2)
        return lon - hlam, lat - hphi, lon + hlam, lat + hphi

    @property
    def geometry(self) -> Polygon:
        return box(*self.bounds_deg)

    @property
    def key(self) -> tuple[int, int]:
        return (self.m, self.z)


def build_grid(region_polygon_deg: Polygon, tile_size_m: float) -> dict[tuple[int, int], Tile]:
    """Cover `region_polygon_deg` (lon/lat degrees) with tiles of size `tile_size_m`.

    Tiles partially overlapping the polygon are retained (base paper, Fig. 5).
    """
    corners = np.asarray(region_polygon_deg.exterior.coords[:-1])
    lat0, lon0 = spherical_centroid(np.radians(corners[:, 1]), np.radians(corners[:, 0]))
    dphi, dlam = angular_spans(tile_size_m, lat0)

    minx, miny, maxx, maxy = region_polygon_deg.bounds
    m_lo = int(np.floor((np.radians(miny) - lat0) / dphi)) - 1
    m_hi = int(np.ceil((np.radians(maxy) - lat0) / dphi)) + 1
    z_lo = int(np.floor((np.radians(minx) - lon0) / dlam)) - 1
    z_hi = int(np.ceil((np.radians(maxx) - lon0) / dlam)) + 1

    tiles: dict[tuple[int, int], Tile] = {}
    for m in range(m_lo, m_hi + 1):
        for z in range(z_lo, z_hi + 1):
            t = Tile(m, z, lat0 + m * dphi, lon0 + z * dlam, dphi, dlam)
            if t.geometry.intersects(region_polygon_deg):
                tiles[t.key] = t
    return tiles


def assign_structures(
    tiles: dict[tuple[int, int], Tile], structures: gpd.GeoDataFrame
) -> dict[tuple[int, int], Tile]:
    """Clip every structure to each overlapping tile (split-at-boundary, Fig. 7).

    `structures` must be EPSG:4326 with at least columns: geometry, height (m),
    kind in {building, vegetation, water}.
    """
    sindex = structures.sindex
    for tile in tiles.values():
        idx = list(sindex.query(tile.geometry, predicate="intersects"))
        if not idx:
            tile.structures = structures.iloc[0:0].copy()
            continue
        sub = structures.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(tile.geometry)
        sub = sub[~sub.geometry.is_empty]
        tile.structures = sub.reset_index(drop=True)
    return tiles


def tiles_to_gdf(tiles: dict[tuple[int, int], Tile]) -> gpd.GeoDataFrame:
    rows = [{"m": t.m, "z": t.z, "geometry": t.geometry} for t in tiles.values()]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")
