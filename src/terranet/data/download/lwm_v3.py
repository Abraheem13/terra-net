"""DeepMIMOv3-format (.mat) city scenarios -> measurements.parquet.

Verified against huggingface.co/datasets/wi-lab/lwm, city_6_miami:
  BS<k>_UE_0-42984.mat: channels (n_ue,) object array of mat_struct with field
  `p` of shape (8, n_paths); rx_locs (n_ue, 5); tx_loc (3,).

Row layout of `p`, confirmed empirically:
  0 phase(deg)  1 ToA(s)  2 power(dB)  3 AoD_az  4 AoD_el  5 AoA_az
  6 AoA_el      7 LoS flag(0/1)
rx_locs columns: 0-2 = x,y,z (local metres), 3 = Tx-Rx Euclidean distance (m).

params.mat: carrier_freq 3.5 GHz, transmit_power 0 dBm -> PL = -total_gain_dB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

POWER_ROW, LOS_ROW = 2, 7
M_PER_DEG_LAT = 111_320.0


def load_bs(mat_path: Path):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    ch, rx, tx = m["channels"], np.asarray(m["rx_locs"], float), np.asarray(m["tx_loc"], float)
    pl = np.full(len(ch), np.nan)
    los = np.zeros(len(ch), bool)
    for i, c in enumerate(ch):
        p = getattr(c, "p", None)
        if p is None or np.size(p) == 0:
            continue
        p = np.asarray(p, float)
        p = p.reshape(8, -1) if p.ndim == 1 else p
        gain = 10.0 * np.log10(np.power(10.0, p[POWER_ROW] / 10.0).sum())
        pl[i] = -gain
        los[i] = bool(p[LOS_ROW, 0])
    return pl, los, rx, tx


def convert(scenario_dir: Path, city: str, out_root: Path,
            origin_lat: float, origin_lon: float) -> Path:
    params = sio.loadmat(scenario_dir / "params.mat", squeeze_me=True)
    f_ghz = float(params["carrier_freq"]) / 1e9
    p_tx = float(params["transmit_power"])

    frames = []
    for mat in sorted(scenario_dir.glob("BS*_UE_*.mat")):
        bs_id = mat.name.split("_")[0]
        pl, los, rx, tx = load_bs(mat)
        ok = np.isfinite(pl)
        lat = origin_lat + rx[ok, 1] / M_PER_DEG_LAT
        lon = origin_lon + rx[ok, 0] / (M_PER_DEG_LAT * np.cos(np.radians(origin_lat)))
        frames.append(pd.DataFrame({
            "bs_id": bs_id,
            "rx_lat": lat, "rx_lon": lon, "rx_h": rx[ok, 2],
            "tx_lat": origin_lat + tx[1] / M_PER_DEG_LAT,
            "tx_lon": origin_lon + tx[0] / (M_PER_DEG_LAT * np.cos(np.radians(origin_lat))),
            "tx_h": tx[2],
            "dist_m": rx[ok, 3],
            "pathloss_db": pl[ok] + p_tx,
            "los": los[ok],
        }))
        print(f"  {bs_id}: {ok.sum():6d}/{len(pl)} UEs, "
              f"PL {np.nanmin(pl):.1f}-{np.nanmax(pl):.1f} dB, LoS {los[ok].mean():.3f}")

    df = pd.concat(frames, ignore_index=True)
    out = out_root / city
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "measurements.parquet")
    (out / "origin.json").write_text(json.dumps({
        "source": "wi-lab/lwm (DeepMIMOv3 format)", "scenario": scenario_dir.name,
        "freq_ghz": f_ghz, "tx_power_dbm": p_tx,
        "origin_lat": origin_lat, "origin_lon": origin_lon,
        "origin_is_true_geo": False, "n_measurements": int(len(df)),
        "n_bs": int(df.bs_id.nunique()),
    }, indent=2))
    print(f"[ok] {city}: {len(df)} measurements, {df.bs_id.nunique()} BSs -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--out", default="data/raw/lwm")
    p.add_argument("--origin-lat", type=float, required=True)
    p.add_argument("--origin-lon", type=float, required=True)
    a = p.parse_args()
    convert(Path(a.scenario_dir), a.city, Path(a.out), a.origin_lat, a.origin_lon)
