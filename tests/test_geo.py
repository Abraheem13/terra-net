import numpy as np

from terranet.utils.geo import angular_spans, haversine_m, spherical_centroid


def test_centroid_square():
    lat = np.radians([51.0, 51.0, 51.1, 51.1])
    lon = np.radians([-0.1, 0.0, 0.0, -0.1])
    lat0, lon0 = spherical_centroid(lat, lon)
    assert abs(np.degrees(lat0) - 51.05) < 1e-3
    assert abs(np.degrees(lon0) - (-0.05)) < 1e-3


def test_angular_span_size():
    dphi, dlam = angular_spans(1000.0, np.radians(51.0))
    # meridional 1000 m arc at any latitude
    assert abs(dphi * 6_371_000 - 1000.0) < 1e-6
    assert dlam > dphi  # zonal span widens away from equator


def test_haversine_symmetry():
    d = haversine_m(np.radians(51.0), np.radians(0.0), np.radians(51.0), np.radians(0.01))
    assert 600 < d < 800  # ~700 m at this latitude
