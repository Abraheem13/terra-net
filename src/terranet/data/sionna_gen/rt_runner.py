"""Sionna RT radio-map generation, version-adaptive.

Supports Sionna 0.19 (`scene.coverage_map`) and Sionna 1.x
(`RadioMapSolver`). The detected API and version are written into the run
metadata so results are reproducible against a specific Sionna release --
important because 1.10+ removed diffraction, which changes NLoS path loss.

Output per city: measurements.parquet with real lat/lon, matching the schema
used by the DeepMIMO/LWM loaders so downstream code is shared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import LocalFrame


def _detect_api():
    import sionna
    ver = getattr(sionna, "__version__", "unknown")
    try:
        from sionna.rt import RadioMapSolver  # noqa: F401
        return "modern", ver
    except Exception:
        return "legacy", ver


def _radio_map_legacy(scene, cell_size, samples):
    cm = scene.coverage_map(cm_cell_size=(cell_size, cell_size), num_samples=samples)
    return np.asarray(cm.path_gain).squeeze(), np.asarray(cm.cell_centers)


def _radio_map_modern(scene, cell_size, samples):
    from sionna.rt import RadioMapSolver
    solver = RadioMapSolver()
    rm = solver(scene, cell_size=(cell_size, cell_size), samples_per_tx=samples)
    return np.asarray(rm.path_gain).squeeze(), np.asarray(rm.cell_centers)


def generate_city(scene_dir: Path, out_dir: Path, *, freq_ghz: float,
                  n_bs: int, bs_mast_agl: float, ue_agl: float,
                  cell_size: float, max_points_per_bs: int,
                  rt_samples: int = 2_000_000, seed: int = 0) -> dict:
    import sionna.rt as rt

    from .bs_placement import rooftop_sites

    api, ver = _detect_api()
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
    size_m = float(meta["size_m"])

    scene = rt.load_scene(str(scene_dir / "scene.xml"))
    scene.frequency = freq_ghz * 1e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                    polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                    polarization="V")

    bs = rooftop_sites(scene_dir / "buildings.ply", n_bs, size_m, bs_mast_agl, seed=seed)
    rng = np.random.default_rng(seed)
    frames = []

    for i, pos in enumerate(bs):
        for name in list(scene.transmitters):
            scene.remove(name)
        scene.add(rt.Transmitter(name=f"bs{i}", position=[float(v) for v in pos]))
        pg, centers = (_radio_map_modern if api == "modern" else _radio_map_legacy)(
            scene, cell_size, rt_samples)

        cc = centers.reshape(-1, 3)
        gain_db = 10.0 * np.log10(np.maximum(np.atleast_2d(pg).reshape(-1), 1e-30))
        ok = np.isfinite(gain_db) & (gain_db > -250.0)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            print(f"    bs{i}: no coverage, skipped")
            continue
        if idx.size > max_points_per_bs:
            idx = rng.choice(idx, max_points_per_bs, replace=False)

        x, y = cc[idx, 0], cc[idx, 1]
        lat, lon = frame.to_geo(x, y)
        bs_lat, bs_lon = frame.to_geo(pos[0], pos[1])
        dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (ue_agl - pos[2]) ** 2)
        frames.append(pd.DataFrame({
            "bs_id": f"bs{i}",
            "rx_lat": lat, "rx_lon": lon, "rx_h": ue_agl,
            "tx_lat": float(bs_lat), "tx_lon": float(bs_lon), "tx_h": float(pos[2]),
            "dist_m": dist,
            "pathloss_db": -gain_db[idx],
            "los": np.nan,
        }))
        pl_i = -gain_db[idx]
        print(f"    bs{i}: {len(idx)} points, PL "
              f"{pl_i.min():.1f}-{pl_i.max():.1f} dB")

    if not frames:
        raise RuntimeError(f"{scene_dir.name}: no coverage from any BS")
    df = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "measurements.parquet")
    run = {
        "source": "sionna_rt", "sionna_version": ver, "api": api,
        "freq_ghz": freq_ghz, "cell_size_m": cell_size, "rt_samples": rt_samples,
        "n_bs": int(df.bs_id.nunique()), "n_measurements": int(len(df)),
        "origin_is_true_geo": True, "scene_meta": meta,
    }
    (out_dir / "origin.json").write_text(json.dumps(run, indent=2))
    print(f"  [ok] {len(df)} measurements from {df.bs_id.nunique()} BSs")
    return run
