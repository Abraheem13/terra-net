"""Spherical geodesy helpers (base paper Appendix A, Eqs. A.1-A.6)."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def spherical_centroid(lat_rad: np.ndarray, lon_rad: np.ndarray) -> tuple[float, float]:
    """Centroid of polygon corners on the unit sphere (Eqs. A.1-A.3)."""
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    xm, ym, zm = x.mean(), y.mean(), z.mean()
    lat0 = np.arctan2(zm, np.hypot(xm, ym))
    lon0 = np.arctan2(ym, xm)
    return float(lat0), float(lon0)


def angular_spans(tile_size_m: float, lat0_rad: float) -> tuple[float, float]:
    """(dphi, dlambda) in radians for a tile of `tile_size_m` metres (Eqs. A.5-A.6)."""
    dphi = tile_size_m / EARTH_RADIUS_M
    dlam = tile_size_m / (EARTH_RADIUS_M * abs(np.cos(lat0_rad)))
    return dphi, dlam


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres. Inputs in radians; broadcasts."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
