import numpy as np
import pytest

pytest.importorskip("pyproj")
from terranet.data.sionna_gen.geo import LocalFrame  # noqa: E402


def test_roundtrip_local_geo():
    f = LocalFrame.from_center(52.3676, 4.9041)
    x = np.array([0.0, 500.0, -1200.0])
    y = np.array([0.0, -800.0, 300.0])
    lat, lon = f.to_geo(x, y)
    xx, yy = f.to_local(lon, lat)
    assert np.allclose(xx, x, atol=1e-3)
    assert np.allclose(yy, y, atol=1e-3)


def test_origin_maps_to_center():
    f = LocalFrame.from_center(41.3851, 2.1734)
    lat, lon = f.to_geo(0.0, 0.0)
    assert abs(float(lat) - 41.3851) < 1e-6
    assert abs(float(lon) - 2.1734) < 1e-6


def test_bbox_covers_all_corners():
    """bbox must contain every scene corner; UTM convergence makes the
    lat/lon box slightly larger than the metric square, which is correct."""
    f = LocalFrame.from_center(51.5074, -0.1278)
    w, s, e, n = f.bbox_deg(3000.0)
    h = 1500.0
    for x in (-h, h):
        for y in (-h, h):
            lat, lon = f.to_geo(x, y)
            assert w <= float(lon) <= e
            assert s <= float(lat) <= n
    ns_m = (n - s) * 111_320
    assert 3000 <= ns_m < 3300          # covers, with modest convergence margin


def test_utm_zone_selection():
    assert LocalFrame.from_center(51.5, -0.13).epsg == 32630   # zone 30N
    assert LocalFrame.from_center(40.75, -73.99).epsg == 32618  # zone 18N
    assert LocalFrame.from_center(-33.87, 151.21).epsg == 32756  # southern
