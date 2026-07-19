"""Full 116-dim descriptor extraction per tile + feature-wise normalization (Eq. 5).

6 topographic + 38 built/land (32 built + 6 land-cover) + 72 climatic = 116.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from terranet.data.tiling import Tile

from .built_env import BUILT_NAMES, LAND_NAMES, built_env_features, landcover_features
from .climate import CLIMATE_NAMES, climate_features
from .topographic import TOPO_NAMES, topographic_features

DESCRIPTOR_NAMES = TOPO_NAMES + BUILT_NAMES + LAND_NAMES + CLIMATE_NAMES
assert len(DESCRIPTOR_NAMES) == 116, len(DESCRIPTOR_NAMES)

GROUP_SLICES = {  # for missing-modality robustness experiments
    "topographic": slice(0, 6),
    "built": slice(6, 38),
    "landcover": slice(38, 44),
    "climate": slice(44, 116),
}


class DescriptorExtractor:
    def __init__(self, dem_sampler, climate_cube: xr.Dataset | None, tile_size_m: float):
        self.dem_sampler = dem_sampler  # callable(bounds_deg) -> (asl_samples, agl_samples)
        self.climate_cube = climate_cube
        self.norm_area = tile_size_m ** 2

    def extract(self, tile: Tile) -> np.ndarray:
        area = self.norm_area  # near-equal-area tiling
        asl, agl = self.dem_sampler(tile.bounds_deg)
        s = tile.structures
        b = s[s["kind"] == "building"] if s is not None else s
        w = s[s["kind"] == "water"] if s is not None else s
        v = s[s["kind"] == "vegetation"] if s is not None else s
        parts = [
            topographic_features(asl, agl, area, self.norm_area),
            built_env_features(b, area),
            landcover_features(w, v, area),
        ]
        if self.climate_cube is not None:
            parts.append(climate_features(
                self.climate_cube, float(np.degrees(tile.lat_c)), float(np.degrees(tile.lon_c))
            ))
        else:
            parts.append(np.zeros(72, dtype=np.float32))
        vec = np.concatenate(parts)
        assert vec.shape == (116,)
        return vec


def normalize_featurewise(D: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Base paper Eq. (5): divide each column by its max. Returns (D_norm, col_max).
    Store col_max from TRAINING cities only and reuse on deployment cities."""
    col_max = np.abs(D).max(axis=0)
    return D / (col_max + eps), col_max
