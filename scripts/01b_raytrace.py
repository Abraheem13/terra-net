#!/usr/bin/env python
"""Ray-trace each scene with Sionna RT -> measurements.parquet + origin.json.

Uses the GPU when available, otherwise the CPU (LLVM) backend. All settings
come from the city config. Resumable: skips cities already generated.
"""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from terranet.data.sionna_gen.rt_runner import generate_city


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_cities.yaml")
    ap.add_argument("--city", default=None)
    ap.add_argument("--scenes", default="data/raw/sionna_scenes")
    ap.add_argument("--out", default=None, help="default: data/raw/<dataset>")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    cities = cfg.get("rt_cities", None) or list(cfg.cities)
    for city in cities:
        if args.city and city != args.city:
            continue
        scene_dir = Path(args.scenes) / city
        out_dir = Path(args.out or f"data/raw/{cfg.dataset}") / city
        if not (scene_dir / "scene.xml").exists():
            print(f"[skip] {city}: scene not built")
            continue
        if (out_dir / "measurements.parquet").exists():
            print(f"[skip] {city}: already ray-traced")
            continue
        print(f"[rt] {city}")
        try:
            generate_city(scene_dir, out_dir, freq_ghz=float(cfg.freq_ghz),
                          n_bs=int(cfg.bs_per_city),
                          bs_mast_agl=float(cfg.bs_height_agl),
                          ue_agl=float(cfg.ue_height_agl),
                          cell_size=float(cfg.cell_size_m),
                          rt_samples=int(cfg.rt_samples), max_depth=int(cfg.max_depth),
                          max_pathloss_db=float(cfg.max_pathloss_db),
                          siting=str(cfg.get("siting", "tallest")))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
