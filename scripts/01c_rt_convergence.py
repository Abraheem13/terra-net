#!/usr/bin/env python
"""Ray-budget convergence of the Monte-Carlo radio-map solver.

For one site of one city, radio maps are computed with increasing numbers of
rays and compared with the largest budget: number of cells reached below
several path-loss levels, and RMS / mean difference of the path loss of cells
reached by both. Justifies the budget used in 01b_raytrace.py.

Output: outputs/tables/rt_convergence.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from terranet.data.sionna_gen.bs_placement import rooftop_sites
from terranet.data.sionna_gen.rt_runner import select_variant

BUDGETS = (10 ** 7, 3 * 10 ** 7, 10 ** 8, 3 * 10 ** 8)
LEVELS = (120.0, 130.0, 140.0, 150.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_cities.yaml")
    ap.add_argument("--city", default="amsterdam")
    ap.add_argument("--scenes", default="data/raw/sionna_scenes")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    select_variant()
    import mitsuba as mi
    import sionna.rt as rt

    d = Path(args.scenes) / args.city
    scene = rt.load_scene(str(d / "scene.xml"))
    scene.frequency = float(cfg.freq_ghz) * 1e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    pos = rooftop_sites(d / "buildings.ply", int(cfg.bs_per_city), float(cfg.size_m),
                        float(cfg.bs_height_agl), seed=0)[0]
    scene.add(rt.Transmitter(name="bs00", position=[float(v) for v in pos]))
    solver = rt.RadioMapSolver()
    maps, secs = {}, {}
    for n in BUDGETS:
        t = time.time()
        rm = solver(scene, center=mi.Point3f(0.0, 0.0, float(cfg.ue_height_agl)),
                    orientation=mi.Point3f(0.0, 0.0, 0.0),
                    size=mi.Point2f(float(cfg.size_m), float(cfg.size_m)),
                    cell_size=mi.Point2f(float(cfg.cell_size_m), float(cfg.cell_size_m)),
                    samples_per_tx=n, seed=0, max_depth=int(cfg.max_depth), los=True,
                    specular_reflection=True, diffuse_reflection=True, refraction=True,
                    diffraction=True, edge_diffraction=True)
        with np.errstate(divide="ignore"):
            maps[n] = -10 * np.log10(np.asarray(rm.path_gain)[0])
        secs[n] = time.time() - t
        print(f"{n:.0e} rays: {secs[n]:.1f} s", flush=True)
    ref = maps[BUDGETS[-1]]
    rows = []
    for n in BUDGETS:
        r = {"rays": n, "seconds": round(secs[n], 1)}
        for L in LEVELS:
            a = maps[n] <= L
            both = a & (ref <= L)
            r[f"cells_le_{int(L)}"] = int(a.sum())
            r[f"frac_of_ref_{int(L)}"] = float(a.sum() / max((ref <= L).sum(), 1))
            r[f"rms_diff_{int(L)}"] = float(np.sqrt(((maps[n] - ref)[both] ** 2).mean()))
            r[f"bias_{int(L)}"] = float((maps[n] - ref)[both].mean())
        rows.append(r)
    out = Path("outputs/tables/rt_convergence.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
