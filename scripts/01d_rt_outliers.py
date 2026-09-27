#!/usr/bin/env python
"""Source of the below-free-space outliers of the radio-map estimator.

Finds the site of the corpus with the most links more than `qc_fspl_margin_db`
below free space, re-traces it with the corpus settings once as is and once
with diffraction disabled, and counts the cells
more than 6 and 10 dB below free space in each run.

Output: outputs/tables/rt_outliers.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from terranet.data.labels import free_space_db, link_distance_m
from terranet.data.sionna_gen.rt_runner import select_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_cities.yaml")
    ap.add_argument("--analysis", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    gen, acfg = OmegaConf.load(args.config), OmegaConf.load(args.analysis)
    base = OmegaConf.load("configs/base.yaml")
    raw = Path(base.paths.raw)
    margin = float(acfg.qc_fspl_margin_db)
    worst = None
    for city in gen.cities:
        m = pd.read_parquet(raw / gen.dataset / city / "measurements.parquet")
        low = m.pathloss_db < free_space_db(link_distance_m(m), float(gen.freq_ghz)) - margin
        cnt = low.groupby(m.bs_id).sum()
        if worst is None or cnt.max() > worst[2]:
            worst = (city, cnt.idxmax(), int(cnt.max()))
    city, bs_id, n_low = worst
    site = int(bs_id[2:])

    select_variant()
    import mitsuba as mi
    import sionna.rt as rt

    from terranet.data.sionna_gen.bs_placement import rooftop_sites
    scene_dir = raw / "sionna_scenes" / city
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    scene = rt.load_scene(str(scene_dir / "scene.xml"))
    scene.frequency = float(gen.freq_ghz) * 1e9
    for a in ("tx_array", "rx_array"):
        setattr(scene, a, rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                         polarization="V"))
    pos = rooftop_sites(scene_dir / "buildings.ply", int(gen.bs_per_city), float(meta["size_m"]),
                        float(gen.bs_height_agl), seed=0)[site]
    scene.add(rt.Transmitter(name="tx", position=[float(v) for v in pos]))
    base_prop = dict(max_depth=int(gen.max_depth), los=True, specular_reflection=True,
                     diffuse_reflection=True, refraction=True, diffraction=True,
                     edge_diffraction=True)
    rows = []
    for name, change in (("corpus settings", {}),
                         ("diffraction off", dict(diffraction=False, edge_diffraction=False))):
        rm = rt.RadioMapSolver()(
            scene, center=mi.Point3f(0.0, 0.0, float(gen.ue_height_agl)),
            orientation=mi.Point3f(0.0, 0.0, 0.0),
            size=mi.Point2f(float(meta["size_m"]), float(meta["size_m"])),
            cell_size=mi.Point2f(float(gen.cell_size_m), float(gen.cell_size_m)),
            samples_per_tx=int(gen.rt_samples), seed=site, **{**base_prop, **change})
        pg = np.asarray(rm.path_gain)[0].reshape(-1)
        cc = np.asarray(rm.cell_centers).reshape(-1, 3)
        with np.errstate(divide="ignore"):
            pl = -10 * np.log10(pg)
        ok = np.isfinite(pl) & (pl <= float(gen.max_pathloss_db))
        ex = pl[ok] - free_space_db(np.linalg.norm(cc[ok] - pos, axis=1), float(gen.freq_ghz))
        rows.append(dict(city=city, site=bs_id, corpus_links_below_margin=n_low, run=name,
                         rays=int(gen.rt_samples), cells=int(ok.sum()),
                         below_6db=int((ex < -6).sum()),
                         below_10db=int((ex < -10).sum()), min_excess_db=float(ex.min())))
        print(rows[-1], flush=True)
    out = Path(base.paths.outputs) / "tables" / "rt_outliers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
