import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box  # noqa: E402

from terranet.data.tiling import build_grid, tiles_to_gdf  # noqa: E402


def test_grid_covers_region():
    region = box(-0.05, 51.50, 0.00, 51.53)
    tiles = build_grid(region, 500.0)
    gdf = tiles_to_gdf(tiles)
    uncovered = region.difference(gdf.union_all().buffer(1e-9))
    assert uncovered.area < 1e-10 * region.area


def test_neighbor_indexing():
    region = box(-0.02, 51.50, 0.02, 51.52)
    tiles = build_grid(region, 500.0)
    keys = set(tiles)
    m0 = [k for k in keys if k == (0, 0)]
    assert m0, "central tile must exist"
    # eastern neighbour center is dlam east of centre
    t00, t01 = tiles[(0, 0)], tiles.get((0, 1))
    if t01:
        assert np.isclose(t01.lon_c - t00.lon_c, t00.dlam)


def test_point_index_matches_geometry():
    from shapely.geometry import Point

    from terranet.data.tiling import grid_frame, point_tile_index
    region = box(4.88, 52.36, 4.92, 52.38)
    g = grid_frame(region, 100.0)
    rng = np.random.default_rng(0)
    lon = rng.uniform(4.881, 4.919, 400)
    lat = rng.uniform(52.361, 52.379, 400)
    idx = point_tile_index(g, lon, lat)
    assert (idx >= 0).all()
    for i in range(0, 400, 7):
        r = g.iloc[idx[i]]
        assert box(r.minx, r.miny, r.maxx, r.maxy).buffer(1e-12).contains(Point(lon[i], lat[i]))
