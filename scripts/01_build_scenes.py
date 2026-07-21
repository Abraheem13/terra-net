#!/usr/bin/env python
"""Phase 1a (CPU): build Mitsuba scenes from OSM for every city.

Network + CPU bound; safe to rerun (OSM layers are cached as GeoJSON).
No Sionna needed here, so it runs in the main env.
"""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from terranet.data.sionna_gen.osm_scene import build_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_cities.yaml")
    ap.add_argument("--city", default=None)
    ap.add_argument("--out", default="data/raw/sionna_scenes")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    summary = []
    for city, (lat, lon) in cfg.cities.items():
        if args.city and city != args.city:
            continue
        out = Path(args.out) / city
        if (out / "scene_meta.json").exists():
            print(f"[skip] {city} (already built)")
            summary.append(json.loads((out / "scene_meta.json").read_text()))
            continue
        print(f"[build] {city}")
        try:
            meta = build_scene(city, float(lat), float(lon), float(cfg.size_m), out)
            p = meta["height_provenance"]
            print(f"  {p['n_buildings']} buildings | height tag {p['frac_height_tag']:.0%} "
                  f"levels {p['frac_levels']:.0%} imputed {p['frac_imputed']:.0%} "
                  f"| water {meta['n_water']} veg {meta['n_vegetation']}")
            summary.append(meta)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    if summary:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n{len(summary)} scenes ready")


if __name__ == "__main__":
    main()
