#!/usr/bin/env python
"""Ray-budget convergence of the Monte-Carlo radio-map solver.

For the given sites of the given cities, radio maps are computed with
increasing numbers of rays and compared with the largest budget (the
reference): number of cells reached below several path-loss levels, and RMS /
mean difference of the path loss of cells reached by both. Justifies the
budget used in 01b_raytrace.py and gives the solver noise floor at that
budget. Rows are appended per (city, site); an existing row is replaced.

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

BUDGETS = (10 ** 7, 3 * 10 ** 7, 10 ** 8, 3 * 10 ** 8, 10 ** 9)
LEVELS = (120.0, 130.0, 140.0, 150.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_cities.yaml")
    ap.add_argument("--city", nargs="+", default=["amsterdam"])
    ap.add_argument("--site", nargs="+", type=int, default=[0])
    ap.add_argument("--scenes", default="data/raw/sionna_scenes")
    ap.add_argument("--max-rays", type=int, default=BUDGETS[-1])
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    select_variant()
    import mitsuba as mi
    import sionna.rt as rt

    out = Path("outputs/tables/rt_convergence.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    budgets = [b for b in BUDGETS if b <= args.max_rays]
    rows = []
    for city in args.city:
        d = Path(args.scenes) / city
        scene = rt.load_scene(str(d / "scene.xml"))
        scene.frequency = float(cfg.freq_ghz) * 1e9
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        sites = rooftop_sites(d / "buildings.ply", int(cfg.bs_per_city), float(cfg.size_m),
                              float(cfg.bs_height_agl), seed=0)
        solver = rt.RadioMapSolver()
        for site in args.site:
            for name in list(scene.transmitters):
                scene.remove(name)
            scene.add(rt.Transmitter(name=f"bs{site:02d}",
                                     position=[float(v) for v in sites[site]]))
            maps, secs = {}, {}
            for n in budgets:
                t = time.time()
                rm = solver(scene, center=mi.Point3f(0.0, 0.0, float(cfg.ue_height_agl)),
                            orientation=mi.Point3f(0.0, 0.0, 0.0),
                            size=mi.Point2f(float(cfg.size_m), float(cfg.size_m)),
                            cell_size=mi.Point2f(float(cfg.cell_size_m), float(cfg.cell_size_m)),
                            samples_per_tx=n, seed=site, max_depth=int(cfg.max_depth), los=True,
                            specular_reflection=True, diffuse_reflection=True, refraction=True,
                            diffraction=True, edge_diffraction=True)
                with np.errstate(divide="ignore"):
                    maps[n] = -10 * np.log10(np.asarray(rm.path_gain)[0])
                secs[n] = time.time() - t
                print(f"{city} site {site} {n:.0e} rays: {secs[n]:.1f} s", flush=True)
            ref = maps[budgets[-1]]
            for n in budgets:
                r = {"city": city, "site": site, "site_h_m": float(sites[site][2]),
                     "ref_rays": budgets[-1], "rays": n, "seconds": round(secs[n], 1)}
                for L in LEVELS:
                    a = maps[n] <= L
                    both = a & (ref <= L)
                    r[f"cells_le_{int(L)}"] = int(a.sum())
                    r[f"frac_of_ref_{int(L)}"] = float(a.sum() / max((ref <= L).sum(), 1))
                    r[f"rms_diff_{int(L)}"] = float(np.sqrt(((maps[n] - ref)[both] ** 2).mean()))
                    r[f"bias_{int(L)}"] = float((maps[n] - ref)[both].mean())
                rows.append(r)
            df = pd.DataFrame(rows)
            if len(old):
                keep = ~(old.city.isin(df.city) & old.site.isin(df.site)) if "city" in old \
                    else pd.Series(False, index=old.index)
                df = pd.concat([old[keep], df], ignore_index=True)
            df.to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
