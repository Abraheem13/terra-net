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
