"""DeepMIMO v4 scenario -> measurements.parquet mapper.

Verified against deepmimo==4.0.3, scenario asu_campus_3p5:
  keys: power (n_rx, max_paths) per-path gain in dB (NaN = no path),
        delay, rx_pos (n_rx, 3), tx_pos (3,) or (1,3), inter (interaction
        codes; a path with zero interactions = LoS), scene, rt_params.

Path loss per RX = -10*log10( sum_paths 10^(power/10) ), NaN-safe.
Positions are LOCAL metres; we map to pseudo lat/lon around a per-city
origin so the tiling pipeline runs unchanged. The origin is stored in
origin.json — replace with the true geo origin (from scene/rt_params or
deepmimo.net scenario page) before extracting OSM-based descriptors.

License: CC BY-NC-SA 4.0 — academic use, cite DeepMIMO.
Usage:
    python -m terranet.data.download.deepmimo --scenario asu_campus_3p5 \
        --city asu --out data/raw/deepmimo --origin-lat 33.4242 --origin-lon -111.9281
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_320.0


def local_to_geo(xy: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + xy[:, 1] / M_PER_DEG_LAT
    lon = lon0 + xy[:, 0] / (M_PER_DEG_LAT * np.cos(np.radians(lat0)))
    return lat, lon


def pathloss_from_paths(power_db: np.ndarray) -> np.ndarray:
    """Total path gain over valid paths; PL = -gain. NaN rows (no paths) -> NaN."""
    lin = np.power(10.0, power_db / 10.0)
    total = np.nansum(lin, axis=1)
    with np.errstate(divide="ignore"):
        gain_db = 10.0 * np.log10(total)
    gain_db[~np.isfinite(gain_db)] = np.nan
    return -gain_db


def los_flags(inter, n_rx: int) -> np.ndarray:
    """LoS if the strongest path has zero interactions. Fallback: all False."""
    try:
        arr = np.asarray(inter)
        if arr.ndim == 2:                      # (n_rx, max_paths) interaction counts/codes
            first = arr[:, 0]
            return (np.nan_to_num(first, nan=-1) == 0)
    except Exception:
        pass
    return np.zeros(n_rx, dtype=bool)


def convert(scenario: str, city: str, out_root: Path,
            origin_lat: float, origin_lon: float) -> Path:
    import deepmimo as dm

    d = dm.load(scenario)
    power = np.asarray(d["power"])
    rx = np.asarray(d["rx_pos"], dtype=float)
    tx = np.asarray(d["tx_pos"], dtype=float).reshape(-1)[:3]

    pl = pathloss_from_paths(power)
    valid = np.isfinite(pl)
    rx_lat, rx_lon = local_to_geo(rx[valid], origin_lat, origin_lon)
    tx_lat, tx_lon = local_to_geo(tx[None, :2] * np.ones((1, 2)), origin_lat, origin_lon)

    df = pd.DataFrame({
        "rx_lat": rx_lat, "rx_lon": rx_lon,
        "tx_lat": np.full(valid.sum(), tx_lat[0]),
        "tx_lon": np.full(valid.sum(), tx_lon[0]),
        "rx_h": rx[valid, 2], "tx_h": tx[2],
        "pathloss_db": pl[valid],
        "los": los_flags(d.get("inter"), len(pl))[valid],
    })
    out = out_root / city
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "measurements.parquet")
    (out / "origin.json").write_text(json.dumps({
        "scenario": scenario, "origin_lat": origin_lat, "origin_lon": origin_lon,
        "origin_is_true_geo": False, "n_rx_total": int(len(pl)),
        "n_rx_valid": int(valid.sum()),
    }, indent=2))
    print(f"[ok] {city}: {valid.sum()}/{len(pl)} valid RX, "
          f"PL range [{np.nanmin(pl):.1f}, {np.nanmax(pl):.1f}] dB -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--out", default="data/raw/deepmimo")
    p.add_argument("--origin-lat", type=float, required=True)
    p.add_argument("--origin-lon", type=float, required=True)
    a = p.parse_args()
    convert(a.scenario, a.city, Path(a.out), a.origin_lat, a.origin_lon)
