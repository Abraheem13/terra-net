"""Topographic descriptors (6): tile area; min/mean/std elevation ASL; mean/std elevation AGL.

Elevation is sampled from a DEM raster (Copernicus DEM GLO-30 or national LiDAR DTM)
at a regular grid of points inside the tile.
"""
from __future__ import annotations

import numpy as np


def topographic_features(dem_samples_asl: np.ndarray, dem_samples_agl: np.ndarray,
                         tile_area_m2: float, norm_area_m2: float) -> np.ndarray:
    f = np.zeros(6, dtype=np.float32)
    f[0] = tile_area_m2 / norm_area_m2
    if dem_samples_asl.size:
        f[1] = np.min(dem_samples_asl)
        f[2] = np.mean(dem_samples_asl)
        f[3] = np.std(dem_samples_asl)
    if dem_samples_agl.size:
        f[4] = np.mean(dem_samples_agl)
        f[5] = np.std(dem_samples_agl)
    return f


TOPO_NAMES = [
    "tile_area", "elev_asl_min", "elev_asl_mean", "elev_asl_std",
    "elev_agl_mean", "elev_agl_std",
]
