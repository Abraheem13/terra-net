"""Sionna RT radio-map generation (Sionna RT >= 1.0, `RadioMapSolver`).

The measurement plane is set explicitly: the full scene square, at receiver
height above the flat ground, with the configured cell size. The Mitsuba
variant is CUDA when a GPU is present and LLVM (CPU) otherwise; the solver
version, variant and every propagation setting are written to origin.json.

Output per city: measurements.parquet (one row per ray-traced cell and site:
serving site, receiver and transmitter lat/lon/height, 3-D distance, path
loss) and origin.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import LocalFrame


def select_variant() -> str:
    import mitsuba as mi
    for v in ("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"):
        if v in mi.variants():
            try:
                mi.set_variant(v)
                return v
            except Exception:  # no GPU / driver: fall through to LLVM
                continue
    raise RuntimeError("no usable Mitsuba variant (install LLVM for the CPU backend)")


def generate_city(scene_dir: Path, out_dir: Path, *, freq_ghz: float, n_bs: int,
                  bs_mast_agl: float, ue_agl: float, cell_size: float,
                  rt_samples: int, max_depth: int, max_pathloss_db: float,
                  seed: int = 0, siting: str = "tallest") -> dict:
    variant = select_variant()
    import mitsuba as mi
    import sionna.rt as rt

    from .bs_placement import rooftop_sites

    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
    size_m = float(meta["size_m"])

    scene = rt.load_scene(str(scene_dir / "scene.xml"))
    scene.frequency = freq_ghz * 1e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    solver = rt.RadioMapSolver()
    prop = dict(max_depth=int(max_depth), los=True, specular_reflection=True,
                diffuse_reflection=True, refraction=True, diffraction=True,
                edge_diffraction=True)

    bs = rooftop_sites(scene_dir / "buildings.ply", n_bs, size_m, bs_mast_agl, seed=seed,
                       policy=siting)
    frames, t0 = [], time.time()
    for i, pos in enumerate(bs):
        for name in list(scene.transmitters):
            scene.remove(name)
        scene.add(rt.Transmitter(name=f"bs{i}", position=[float(v) for v in pos]))
        rm = solver(scene, center=mi.Point3f(0.0, 0.0, float(ue_agl)),
                    orientation=mi.Point3f(0.0, 0.0, 0.0),
                    size=mi.Point2f(size_m, size_m), cell_size=mi.Point2f(cell_size, cell_size),
                    samples_per_tx=int(rt_samples), seed=seed + i, **prop)
        pg = np.asarray(rm.path_gain)[0].reshape(-1)
        cc = np.asarray(rm.cell_centers).reshape(-1, 3)
        with np.errstate(divide="ignore"):
            pl = -10.0 * np.log10(pg)
        ok = np.isfinite(pl) & (pl <= max_pathloss_db)
        x, y = cc[ok, 0], cc[ok, 1]
        lat, lon = frame.to_geo(x, y)
        bs_lat, bs_lon = frame.to_geo(pos[0], pos[1])
        dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (cc[ok, 2] - pos[2]) ** 2)
        frames.append(pd.DataFrame({
            "bs_id": f"bs{i:02d}", "rx_lat": lat, "rx_lon": lon, "rx_h": cc[ok, 2],
            "tx_lat": float(bs_lat), "tx_lon": float(bs_lon), "tx_h": float(pos[2]),
            "dist_m": dist, "pathloss_db": pl[ok]}))
        print(f"    bs{i:02d}: {ok.sum():6d} cells <= {max_pathloss_db:g} dB, "
              f"PL {pl[ok].min():.1f}-{pl[ok].max():.1f} dB", flush=True)

    df = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "measurements.parquet")
    run = {
        "source": "sionna_rt", "sionna_version": rt.__version__, "api": "RadioMapSolver",
        "mitsuba_variant": variant, "freq_ghz": freq_ghz, "cell_size_m": cell_size,
        "rt_samples": int(rt_samples), "n_bs": int(df.bs_id.nunique()),
        "n_measurements": int(len(df)), "max_pathloss_db": max_pathloss_db,
        "ue_height_m": ue_agl, "site_mast_m": bs_mast_agl, "seed": seed, "siting": siting,
        "propagation": prop, "materials": {"buildings": "itu concrete",
                                           "ground": "itu medium_dry_ground"},
        "wall_clock_s": round(time.time() - t0, 1),
        "origin_is_true_geo": True, "scene_meta": meta,
    }
    (out_dir / "origin.json").write_text(json.dumps(run, indent=2))
    print(f"  [ok] {len(df)} links from {df.bs_id.nunique()} sites in {run['wall_clock_s']} s")
    return run
