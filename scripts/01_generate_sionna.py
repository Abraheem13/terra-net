#!/usr/bin/env python
"""Phase 1 (optional, GPU): generate Sionna RT radio maps for European cities.

Run inside the [rt] environment (sionna==0.19.*). One scene + N BS per city.
"""
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from terranet.data.sionna_gen.rt_runner import run_radio_map
from terranet.data.sionna_gen.scene_builder import build_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_gen.yaml")
    ap.add_argument("--city", default=None, help="limit to one city")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    rng = np.random.default_rng(0)
    for city, bbox in cfg.cities.items():
        if args.city and city != args.city:
            continue
        out = Path("data/raw/sionna_euro") / city
        scene = build_scene(tuple(bbox), out)
        # Random BS on rooftops proxy: uniform in bbox at 25 m height
        lon = rng.uniform(bbox[0], bbox[2], cfg.bs_per_city)
        lat = rng.uniform(bbox[1], bbox[3], cfg.bs_per_city)
        bs = np.c_[lon, lat, np.full(cfg.bs_per_city, 25.0)]
        run_radio_map(scene, bs, f_hz=float(cfg.freq_ghz) * 1e9,
                      cell_size=float(cfg.cell_size_m), out_npz=out / "radio_maps.npz")
        print(f"[ok] {city}")


if __name__ == "__main__":
    main()
