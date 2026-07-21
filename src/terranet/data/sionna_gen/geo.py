"""Local metric frame for a scene.

Every scene has a TRUE geographic anchor: a centre (lat, lon) and a local
east-north-up frame in metres centred on it. Ray tracing runs in the local
frame; measurements are written back as real lat/lon so OSM descriptors,
DEM, and climate all line up with the propagation data. This is the property
the DeepMIMO/LWM scenarios lack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


@dataclass
class LocalFrame:
    lat0: float
    lon0: float
    epsg: int          # projected CRS used as the intermediate
    e0: float          # easting of the origin
    n0: float          # northing of the origin

    @classmethod
    def from_center(cls, lat0: float, lon0: float) -> "LocalFrame":
        zone = int((lon0 + 180.0) // 6) + 1
        epsg = (32600 if lat0 >= 0 else 32700) + zone
        tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        e0, n0 = tf.transform(lon0, lat0)
        return cls(lat0, lon0, epsg, float(e0), float(n0))

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    def to_local(self, lon, lat):
        tf = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        e, n = tf.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(e) - self.e0, np.asarray(n) - self.n0

    def to_geo(self, x, y):
        tf = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(np.asarray(x) + self.e0, np.asarray(y) + self.n0)
        return np.asarray(lat), np.asarray(lon)

    def bbox_deg(self, size_m: float, pad_m: float = 0.0) -> tuple[float, float, float, float]:
        """(west, south, east, north) covering the square scene, in degrees.

        Uses all four corners: UTM grid north differs from true north, so two
        corners would under-cover the scene and OSM would miss features near
        the edges.
        """
        h = size_m / 2.0 + pad_m
        xs = np.array([-h, h, h, -h])
        ys = np.array([-h, -h, h, h])
        lat, lon = self.to_geo(xs, ys)
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())
