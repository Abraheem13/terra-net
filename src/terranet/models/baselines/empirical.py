"""Empirical baselines: Okumura-Hata, COST-231, 3GPP UMa NLOS (TR 38.901).

All return path loss in dB. Frequencies in MHz unless noted.
"""
from __future__ import annotations

import numpy as np


def okumura_hata(d_km: np.ndarray, f_mhz: float, h_b: float = 30.0, h_m: float = 1.5,
                 city: str = "large") -> np.ndarray:
    if city == "large" and f_mhz >= 300:
        a_hm = 3.2 * (np.log10(11.75 * h_m)) ** 2 - 4.97
    else:
        a_hm = (1.1 * np.log10(f_mhz) - 0.7) * h_m - (1.56 * np.log10(f_mhz) - 0.8)
    return (69.55 + 26.16 * np.log10(f_mhz) - 13.82 * np.log10(h_b) - a_hm
            + (44.9 - 6.55 * np.log10(h_b)) * np.log10(np.maximum(d_km, 1e-3)))


def cost231(d_km: np.ndarray, f_mhz: float, h_b: float = 30.0, h_m: float = 1.5,
            metropolitan: bool = True) -> np.ndarray:
    a_hm = (1.1 * np.log10(f_mhz) - 0.7) * h_m - (1.56 * np.log10(f_mhz) - 0.8)
    c = 3.0 if metropolitan else 0.0
    return (46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(h_b) - a_hm
            + (44.9 - 6.55 * np.log10(h_b)) * np.log10(np.maximum(d_km, 1e-3)) + c)


def tr38901_uma_nlos(d3d_m: np.ndarray, f_ghz: float, h_ut: float = 1.5) -> np.ndarray:
    pl = 13.54 + 39.08 * np.log10(np.maximum(d3d_m, 1.0)) + 20.0 * np.log10(f_ghz) \
         - 0.6 * (h_ut - 1.5)
    return pl
