"""OSM footprints -> extruded 3-D meshes -> Mitsuba scene for Sionna RT.

Height policy, in order: OSM `height` tag; else `building:levels` x 3.2 m;
else a per-city median fallback. The policy and the fraction of buildings
relying on each source are recorded in scene_meta.json so the paper can state
exactly how much geometry is measured vs imputed.
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

OSM_TAGS = {
    "building": ({"building": True}, "building"),
    "water": ({"natural": "water", "waterway": ["riverbank", "dock"],
               "landuse": "reservoir"}, "water"),
    "vegetation": ({"landuse": ["grass", "forest", "meadow", "village_green"],
                    "leisure": ["park", "garden"], "natural": ["wood", "scrub"]},
                   "vegetation"),
}


def fetch_layers(bbox: tuple[float, float, float, float], cache_dir: Path,
                 timeout: int = 600) -> dict[str, gpd.GeoDataFrame]:
    """Fetch (or load cached) OSM layers. Cached GeoJSON makes reruns offline."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox = None                       # imported lazily: cached scenes work offline

    out: dict[str, gpd.GeoDataFrame] = {}
    for key, (tags, _) in OSM_TAGS.items():
        f = cache_dir / f"{key}.geojson"
        if f.exists():
            out[key] = gpd.read_file(f)
            continue
        try:
            if ox is None:
                import osmnx as ox_mod
                ox = ox_mod
                ox.settings.requests_timeout = timeout
                ox.settings.use_cache = True
                ox.settings.cache_folder = str(cache_dir / "_osmnx")
            g = ox.features_from_bbox(bbox=bbox, tags=tags)
            g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            keep = ["geometry"] + [c for c in ("height", "building:levels", "building")
                                   if c in g.columns]
            g = g[keep].reset_index(drop=True)
            g.to_file(f, driver="GeoJSON")
            out[key] = g
        except Exception as e:                       # no features is a valid answer
            print(f"    {key}: {type(e).__name__}: {e}")
            out[key] = gpd.GeoDataFrame({"geometry": []}, geometry="geometry",
                                        crs="EPSG:4326")
    return out


def resolve_heights(g: gpd.GeoDataFrame) -> tuple[np.ndarray, dict]:
    n = len(g)
    h = pd.to_numeric(g["height"], errors="coerce") if "height" in g else pd.Series([np.nan] * n)
    lv = (pd.to_numeric(g["building:levels"], errors="coerce") * LEVEL_HEIGHT_M
          if "building:levels" in g else pd.Series([np.nan] * n))
    h = h.reset_index(drop=True)
    lv = lv.reset_index(drop=True)
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
                out_dir: Path) -> dict:
    """Write buildings.ply + ground.ply + scene.xml + scene_meta.json."""
    import trimesh

    frame = LocalFrame.from_center(lat0, lon0)
    bbox = frame.bbox_deg(size_m)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = fetch_layers(bbox, out_dir)

    b = layers["building"]
    b = b[b.geometry.notna()].reset_index(drop=True)
    heights, prov = resolve_heights(b)

    clip = box(-size_m / 2, -size_m / 2, size_m / 2, size_m / 2)
    meshes, kept, failed = [], 0, 0
    for geom, h in zip(b.geometry, heights):
        for poly in _polys(geom):
            x, y = frame.to_local(*np.asarray(poly.exterior.coords).T)
            p = Polygon(np.c_[x, y]).buffer(0).intersection(clip)
            for q in _polys(p):
                if q.area < 4.0:                      # drop slivers
                    continue
                try:
                    meshes.append(trimesh.creation.extrude_polygon(q, float(h)))
                    kept += 1
                except Exception as e:
                    failed += 1
                    if failed == 1:      # surface the cause once, don't hide it
                        print(f"    extrusion error ({type(e).__name__}): {e}")
    if not meshes:
        raise RuntimeError(
            f"{city}: no extrudable buildings ({failed} extrusion failures). "
            "If this is a triangulation error, install: pip install mapbox-earcut")
    buildings = trimesh.util.concatenate(meshes)
    buildings.export(out_dir / "buildings.ply")

    ground = trimesh.creation.box(extents=[size_m * 1.2, size_m * 1.2, 0.2])
    ground.apply_translation([0, 0, -0.1])
    ground.export(out_dir / "ground.ply")

    _write_xml(out_dir)

    meta = {
        "city": city, "lat0": lat0, "lon0": lon0, "size_m": size_m,
        "epsg": frame.epsg, "bbox_wsen": list(bbox),
        "origin_is_true_geo": True,
        "n_building_parts_meshed": kept,
        "n_extrusion_failures": failed,
        "height_provenance": prov,
        "n_water": int(len(layers["water"])),
        "n_vegetation": int(len(layers["vegetation"])),
    }
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


_XML = """<scene version="2.1.0">
  <default name="spp" value="64"/>
  <bsdf type="diffuse" id="mat-itu_concrete">
    <rgb name="reflectance" value="0.35 0.35 0.35"/>
  </bsdf>
  <bsdf type="diffuse" id="mat-itu_medium_dry_ground">
    <rgb name="reflectance" value="0.20 0.20 0.20"/>
  </bsdf>
  <shape type="ply" id="buildings">
    <string name="filename" value="buildings.ply"/>
    <ref id="mat-itu_concrete"/>
  </shape>
  <shape type="ply" id="ground">
    <string name="filename" value="ground.ply"/>
    <ref id="mat-itu_medium_dry_ground"/>
  </shape>
</scene>
"""


def _write_xml(out_dir: Path) -> None:
    (out_dir / "scene.xml").write_text(_XML)
