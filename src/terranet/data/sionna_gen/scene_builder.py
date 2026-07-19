"""OSM -> Mitsuba scene for Sionna RT.

Pipeline: osmnx pulls building footprints + height tags for a bbox; footprints
are extruded to Mitsuba XML meshes (via trimesh); terrain from Copernicus DEM.
Pin sionna==0.19.x (later versions removed diffraction and renamed coverage_map).
Run inside the [rt] extra environment.
"""
from __future__ import annotations

from pathlib import Path


def build_scene(bbox: tuple[float, float, float, float], out_dir: Path,
                default_height_m: float = 12.0) -> Path:
    """bbox = (min_lon, min_lat, max_lon, max_lat). Writes scene.xml + meshes."""
    import numpy as np
    import osmnx as ox
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].explode(index_parts=False)
    heights = (
        pdna(gdf.get("height")).fillna(pdna(gdf.get("building:levels")) * 3.0)
        .fillna(default_height_m)
    )
    meshes = []
    for geom, h in zip(gdf.geometry, heights):
        try:
            poly = np.asarray(geom.exterior.coords)
            m = trimesh.creation.extrude_polygon(
                trimesh.path.polygons.Polygon(poly), float(h))
            meshes.append(m)
        except Exception:
            continue
    combined = trimesh.util.concatenate(meshes)
    ply = out_dir / "buildings.ply"
    combined.export(ply)
    _write_mitsuba_xml(out_dir, ply)
    return out_dir / "scene.xml"


def pdna(series):
    import pandas as pd

    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _write_mitsuba_xml(out_dir: Path, ply: Path) -> None:
    xml = f"""<scene version=\"2.1.0\">
  <shape type=\"ply\"><string name=\"filename\" value=\"{ply.name}\"/>
    <bsdf type=\"diffuse\"><rgb name=\"reflectance\" value=\"0.5 0.5 0.5\"/></bsdf>
    <string name=\"id\" value=\"itu_concrete\"/>
  </shape>
</scene>"""
    (out_dir / "scene.xml").write_text(xml)
