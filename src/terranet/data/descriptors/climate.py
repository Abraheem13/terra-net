"""Climatic descriptors (72): per calendar month, mean & std of
temperature (degC), relative humidity (%), precipitation.

Source: ERA5-Land monthly means via the CDS API (cdsapi). The download script
(scripts/02_extract_descriptors.py --climate) caches a small netCDF per city;
this module only interpolates the cached cube at tile centres.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

VARS = ["t2m", "rh", "tp"]  # temperature, relative humidity (derived), total precip


def climate_features(cube: xr.Dataset, lat_deg: float, lon_deg: float) -> np.ndarray:
    """cube: dims (month: 12x n_years, lat, lon) with variables in VARS.
    Returns 72 values ordered month-major: [m1_t_mean, m1_t_std, m1_rh_mean, ...]."""
    f = np.zeros(72, dtype=np.float32)
    point = cube.interp(latitude=lat_deg, longitude=lon_deg, method="linear")
    for month in range(1, 13):
        sel = point.sel(time=point["time.month"] == month)
        base = (month - 1) * 6
        f[base + 0] = float(sel["t2m"].mean()) - 273.15
        f[base + 1] = float(sel["t2m"].std())
        f[base + 2] = float(sel["rh"].mean())
        f[base + 3] = float(sel["rh"].std())
        f[base + 4] = float(sel["tp"].mean())
        f[base + 5] = float(sel["tp"].std())
    return f


CLIMATE_NAMES = sum(
    [[f"m{m}_t_mean", f"m{m}_t_std", f"m{m}_rh_mean", f"m{m}_rh_std",
      f"m{m}_precip_mean", f"m{m}_precip_std"] for m in range(1, 13)], []
)
