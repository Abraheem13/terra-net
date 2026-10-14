"""The 32 built-environment tile descriptors (the built-structure group of the
descriptor-based framework this study evaluates).

Height classes: <5, 5-10, 10-30, 30-60, >=60 m.
 - footprint-area ratio per class (5)
 - building count per class (5)
 - mean/std footprint area, mean/std height, mean/std volume, footprint density (7)
 - horizontal planes at 5/10/30/60/100 m above ground: number of buildings
   reaching the plane, mean and std of the distance between their centroids (15)
"""
from __future__ import annotations

import numpy as np
from geopandas import GeoDataFrame
from scipy.spatial.distance import pdist

HEIGHT_BINS = [0, 5, 10, 30, 60, np.inf]
PLANES_M = [5, 10, 30, 60, 100]


def built_env_features(buildings: GeoDataFrame, tile_area_m2: float) -> np.ndarray:
    f = np.zeros(32, dtype=np.float32)
    if len(buildings) == 0:
        return f
    h = buildings["height"].to_numpy(float)
    # areas in m^2: reproject to an equal-area CRS once upstream; here `area_m2` column
    a = buildings["area_m2"].to_numpy(float)
    v = a * h

    k = 0
    cls = np.digitize(h, HEIGHT_BINS) - 1
    for c in range(5):  # base-area ratio per class
        f[k] = a[cls == c].sum() / tile_area_m2
        k += 1
    for c in range(5):  # count per class (normalized later feature-wise)
        f[k] = float((cls == c).sum())
        k += 1
    f[k:k + 7] = [a.mean(), a.std(), h.mean(), h.std(), v.mean(), v.std(),
                  a.sum() / tile_area_m2]
    k += 7
    # horizontal-plane intersection metrics
    cent = np.c_[buildings.geometry.centroid.x, buildings.geometry.centroid.y]
    for p in PLANES_M:
        mask = h >= p
        n = int(mask.sum())
        f[k] = n
        if n >= 2:
            d = pdist(cent[mask])  # proxy for LoS inter-building distance at plane p
            f[k + 1], f[k + 2] = d.mean(), d.std()
        k += 3
    return f


BUILT_NAMES = (
    [f"base_area_ratio_h{c}" for c in range(5)]
    + [f"n_buildings_h{c}" for c in range(5)]
    + ["base_area_mean", "base_area_std", "height_mean", "height_std",
       "volume_mean", "volume_std", "building_density"]
    + sum([[f"plane{p}_n", f"plane{p}_dist_mean", f"plane{p}_dist_std"] for p in PLANES_M], [])
)
