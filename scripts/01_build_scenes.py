#!/usr/bin/env python
"""Build the ray-tracing scene of every city from its fetched footprints.

Run scripts/01a_fetch_buildings.py first. CPU only; no Sionna needed.
Skips cities whose scene_meta.json already exists; --rebuild regenerates the
meshes (not versioned) from the versioned footprints and checks that the
number of meshed parts matches the stored metadata.
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
    ap.add_argument("--out", default=None, help="default: data/raw/<scenes_dir> of the config")
    ap.add_argument("--rebuild", action="store_true", help="regenerate meshes of built scenes")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    out_root = Path(args.out or f"data/raw/{cfg.get('scenes_dir', 'sionna_scenes')}")
    materials = list(cfg.get("materials", [])) or None
    cities = cfg.get("rt_cities", None) or list(cfg.cities)

    summary = []
    for city in cities:
        lat, lon = cfg.cities[city]
        if args.city and city != args.city:
            continue
        out = out_root / city
        if (out / "scene_meta.json").exists() and not args.rebuild:
            print(f"[skip] {city} (already built)")
            summary.append(json.loads((out / "scene_meta.json").read_text()))
            continue
        if (out / "scene_meta.json").exists() and (out / "buildings.ply").exists():
            print(f"[skip] {city} (meshes present)")
            summary.append(json.loads((out / "scene_meta.json").read_text()))
            continue
        old = json.loads((out / "scene_meta.json").read_text()) \
            if (out / "scene_meta.json").exists() else None
        print(f"[build] {city}")
        try:
            meta = build_scene(city, float(lat), float(lon), float(cfg.size_m), out,
                               src_dir=Path("data/raw/sionna_scenes") / city,
                               materials=materials)
            n_new, n_old = meta["n_building_parts_meshed"], (old or meta)["n_building_parts_meshed"]
            if n_new != n_old:
                raise RuntimeError(f"{city}: rebuilt mesh has {n_new} parts, metadata says {n_old}")
            if old is not None:                      # keep the versioned metadata untouched
                (out / "scene_meta.json").write_text(json.dumps(old, indent=2))
                meta = old
            p = meta["height_provenance"]
            print(f"  {p['n_buildings']} buildings | height tag {p['frac_height_tag']:.0%} "
                  f"levels {p['frac_levels']:.0%} imputed {p['frac_imputed']:.0%} "
                  f"| {meta['n_building_parts_meshed']} parts meshed")
            summary.append(meta)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    if summary:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n{len(summary)} scenes ready")


if __name__ == "__main__":
    main()
