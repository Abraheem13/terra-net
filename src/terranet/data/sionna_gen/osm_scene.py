"""Building footprints -> extruded 3-D meshes -> Sionna RT scene.

Input: data/raw/sionna_scenes/<city>/building.geojson (scripts/01a_fetch_buildings.py,
Overture Maps buildings, an OpenStreetMap-based open dataset).

Height policy, in order: the record's `height`; else `building:levels` x 3.2 m;
else the per-city median of the resolved heights. The share of buildings
relying on each source is recorded in scene_meta.json so the corpus states
exactly how much geometry is measured vs imputed.

Materials (ITU-R P.2040 models as implemented in Sionna RT): concrete for
buildings, medium-dry ground for the ground plane. With `materials`, every
building part is assigned one of the given ITU materials at random (fixed
seed) and written to one mesh per material (the materials replicate).
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, box

from .geo import LocalFrame

LEVEL_HEIGHT_M = 3.2
FALLBACK_HEIGHT_M = 10.0
MIN_PART_M2 = 4.0


def load_buildings(cache_dir: Path) -> gpd.GeoDataFrame:
    f = cache_dir / "building.geojson"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/01a_fetch_buildings.py first")
    g = gpd.read_file(f)
    return g[g.geometry.notna()].reset_index(drop=True)


def resolve_heights(g: gpd.GeoDataFrame) -> tuple[np.ndarray, dict]:
    n = len(g)
    h = pd.to_numeric(g["height"], errors="coerce") if "height" in g else pd.Series([np.nan] * n)
    lv = (pd.to_numeric(g["building:levels"], errors="coerce") * LEVEL_HEIGHT_M
          if "building:levels" in g else pd.Series([np.nan] * n))
    h = h.reset_index(drop=True).where(lambda v: v > 0)
    lv = lv.reset_index(drop=True).where(lambda v: v > 0)
    from_tag = h.notna()
    from_levels = (~from_tag) & lv.notna()
    merged = h.where(from_tag, lv)
    median = float(merged.median()) if merged.notna().any() else FALLBACK_HEIGHT_M
    final = merged.fillna(median).clip(2.0, 500.0).to_numpy(float)
    prov = {
        "n_buildings": int(n),
        "frac_height_tag": float(from_tag.mean()) if n else 0.0,
        "frac_levels": float(from_levels.mean()) if n else 0.0,
        "frac_imputed": float((~(from_tag | from_levels)).mean()) if n else 0.0,
        "median_height_m": median,
    }
    return final, prov


def _polys(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def build_scene(city: str, lat0: float, lon0: float, size_m: float,
                out_dir: Path, src_dir: Path | None = None,
                materials: list[str] | None = None, seed: int = 0) -> dict:
    """Write buildings.ply + ground.ply + scene.xml + scene_meta.json.

    `src_dir` holds building.geojson (default: out_dir). With `materials`,
    every part gets one of them at random and the meshes are written per
    material (buildings_<material>.ply)."""
    import trimesh

    src_dir = out_dir if src_dir is None else Path(src_dir)
    frame = LocalFrame.from_center(lat0, lon0)
    b = load_buildings(src_dir)
    heights, prov = resolve_heights(b)
    rng = np.random.default_rng(seed)

    clip = box(-size_m / 2, -size_m / 2, size_m / 2, size_m / 2)
    meshes, mats, kept, failed = [], [], 0, 0
    for geom, h in zip(b.geometry, heights, strict=True):
        for poly in _polys(geom):
            x, y = frame.to_local(*np.asarray(poly.exterior.coords).T)
            p = Polygon(np.c_[x, y]).buffer(0).intersection(clip)
            for q in _polys(p):
                if q.area < MIN_PART_M2:                   # drop slivers
                    continue
                try:
                    meshes.append(trimesh.creation.extrude_polygon(q, float(h)))
                    mats.append(materials[rng.integers(len(materials))] if materials
                                else "concrete")
                    kept += 1
                except Exception as e:
                    failed += 1
                    if failed == 1:      # surface the cause once, don't hide it
                        print(f"    extrusion error ({type(e).__name__}): {e}")
    if not meshes:
        raise RuntimeError(f"{city}: no extrudable buildings ({failed} extrusion failures)")
    out_dir.mkdir(parents=True, exist_ok=True)
    shares = {}
    if materials:
        shapes = []
        for m in sorted(set(mats)):
            sel = [mesh for mesh, mm in zip(meshes, mats, strict=True) if mm == m]
            trimesh.util.concatenate(sel).export(out_dir / f"buildings_{m}.ply")
            shares[m] = len(sel) / len(meshes)
            shapes.append(_shape(f"buildings_{m}", f"buildings_{m}.ply", m))
        xml = _xml([*sorted(set(mats)), "medium_dry_ground"], shapes)
    else:
        trimesh.util.concatenate(meshes).export(out_dir / "buildings.ply")
        shares["concrete"] = 1.0
        xml = _xml(["concrete", "medium_dry_ground"],
                   [_shape("buildings", "buildings.ply", "concrete")])

    ground = trimesh.creation.box(extents=[size_m * 1.2, size_m * 1.2, 0.2])
    ground.apply_translation([0, 0, -0.1])
    ground.export(out_dir / "ground.ply")
    (out_dir / "scene.xml").write_text(xml)

    src = json.loads((src_dir / "building_source.json").read_text()) \
        if (src_dir / "building_source.json").exists() else {}
    meta = {
        "city": city, "lat0": lat0, "lon0": lon0, "size_m": size_m,
        "epsg": frame.epsg, "bbox_wsen": list(frame.bbox_deg(size_m)),
        "origin_is_true_geo": True,
        "building_source": src,
        "n_building_parts_meshed": kept,
        "n_extrusion_failures": failed,
        "height_provenance": prov,
        "materials": shares,
    }
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _shape(sid: str, filename: str, material: str) -> str:
    return (f'  <shape type="ply" id="{sid}">\n    <string name="filename" value="{filename}"/>\n'
            f'    <ref id="mat-itu_{material}" name="bsdf"/>\n  </shape>\n')


def _xml(materials: list[str], shapes: list[str]) -> str:
    bsdf = "".join(f'  <bsdf type="itu-radio-material" id="mat-itu_{m}">\n'
                   f'    <string name="type" value="{m}"/>\n'
                   f'    <float name="thickness" value="0.2"/>\n  </bsdf>\n' for m in materials)
    ground = _shape("ground", "ground.ply", "medium_dry_ground")
    return '<scene version="2.1.0">\n' + bsdf + "".join(shapes) + ground + "</scene>\n"
