import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyproj")
from shapely.geometry import Polygon  # noqa: E402

from terranet.data.descriptors.scene_tiles import (  # noqa: E402
    DESCRIPTOR_COLUMNS,
    scene_building_parts,
    tile_descriptors,
)
from terranet.data.sionna_gen.geo import LocalFrame  # noqa: E402
from terranet.data.tiling import grid_frame, region_from_points  # noqa: E402


def _layer(frame):
    # two 40 x 40 m buildings in local metres, one straddling a tile edge
    polys, h = [], []
    for (cx, cy), hh in [((10.0, 10.0), "25"), ((-120.0, 60.0), None)]:
        xs = np.array([-20, 20, 20, -20]) + cx
        ys = np.array([-20, -20, 20, 20]) + cy
        lat, lon = frame.to_geo(xs, ys)
        polys.append(Polygon(np.c_[lon, lat]))
        h.append(hh)
    return gpd.GeoDataFrame({"height": h, "building:levels": ["", "4"]},
                            geometry=polys, crs="EPSG:4326")


def test_area_is_conserved_and_heights_follow_scene_rules():
    frame = LocalFrame.from_center(52.3676, 4.9041)
    parts, heights = scene_building_parts(_layer(frame), frame, 3000.0)
    assert sorted(heights.tolist()) == [12.8, 25.0]          # 4 levels x 3.2 m
    lat, lon = frame.to_geo(np.array([-400.0, 400.0]), np.array([-400.0, 400.0]))
    grid = grid_frame(region_from_points(lon, lat, 0.0), 100.0)
    D = tile_descriptors(grid, parts, heights, frame)
    assert list(D.columns) == DESCRIPTOR_COLUMNS
    from terranet.data.descriptors.scene_tiles import tile_polygons_local
    tile_area = np.array([p.area for p in tile_polygons_local(grid, frame)])
    total = (D["d_building_density"].to_numpy() * tile_area).sum()
    assert abs(total - 2 * 1600.0) / 3200.0 < 0.01
    assert (D["d_n_buildings_h2"] > 0).sum() >= 2           # straddling building split


def test_parts_outside_scene_are_dropped():
    frame = LocalFrame.from_center(52.3676, 4.9041)
    parts, _ = scene_building_parts(_layer(frame), frame, 100.0)
    assert len(parts) == 1                                  # second building is outside
