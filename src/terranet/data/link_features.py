"""Physically motivated features of a transmitter-receiver link.

Computed from the same building geometry as the ray-traced scene (footprint
parts and heights from `scene_tiles.scene_building_parts`), rasterised to a
height map. For every (receiver pixel, site) pair the straight Tx-Rx segment
is sampled and compared with the line-of-sight height profile:

  log_d3, d2        3-D distance (log10 m) and horizontal distance (m)
  tx_h, elev_deg    antenna height and elevation angle seen from the receiver
  blocked_frac      fraction of the segment where a building rises above the
                    line of sight
  n_cross           number of separate buildings the line of sight passes
                    through
  built_len         horizontal length of the segment over building footprints
  max_excess        largest building height above the line of sight (m)
  ke_loss_db        single knife-edge loss J(nu) of the most obstructing point,
                    ITU-R P.526 Eq. (31), nu = h sqrt(2/lambda (1/d1 + 1/d2))
  rx_indoor, rx_bld_h  receiver cell inside a footprint, and that height
  dens_50, hmean_50 built fraction and mean building height in a 100 m square
                    window centred on the receiver
  tx_dens_100       built fraction in a 200 m square window centred on the site

No feature uses path loss, so the features of a target city are available
before any measurement.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

LINK_FEATURES = ["log_d3", "d2", "tx_h", "elev_deg", "blocked_frac", "n_cross", "built_len",
                 "max_excess", "ke_loss_db", "rx_indoor", "rx_bld_h", "dens_50", "hmean_50",
                 "tx_dens_100"]
C = 299_792_458.0


def knife_edge_loss_db(nu: np.ndarray) -> np.ndarray:
    """ITU-R P.526 Eq. (31); zero for nu <= -0.78."""
    j = 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
    return np.where(nu > -0.78, j, 0.0)


class HeightMap:
    """Building-height raster of a scene in its local metric frame."""

    def __init__(self, parts, heights, size_m: float, res_m: float = 2.0):
        from rasterio import features
        from rasterio.transform import from_origin
        self.half, self.res = size_m / 2, res_m
        self.n = int(round(size_m / res_m))
        tr = from_origin(-self.half, self.half, res_m, res_m)
        order = np.argsort(heights)                 # taller parts written last
        shapes = ((parts[i], float(heights[i])) for i in order)
        img = features.rasterize(shapes, out_shape=(self.n, self.n), transform=tr,
                                 fill=0.0, dtype="float32")
        self.h = img[::-1]                          # row 0 = southernmost
        built = (self.h > 0).astype(np.float32)
        self.dens50 = uniform_filter(built, size=int(round(100 / res_m)), mode="constant")
        self.hmean50 = uniform_filter(self.h, size=int(round(100 / res_m)), mode="constant")
        self.dens100 = uniform_filter(built, size=int(round(200 / res_m)), mode="constant")

    def index(self, x, y):
        i = np.clip(((y + self.half) / self.res).astype(np.int64), 0, self.n - 1)
        j = np.clip(((x + self.half) / self.res).astype(np.int64), 0, self.n - 1)
        return i, j

    def at(self, x, y, layer=None):
        i, j = self.index(x, y)
        return (self.h if layer is None else layer)[i, j]


def link_features(hm: HeightMap, px, py, rx_h, sx, sy, sz, freq_hz: float,
                  n_samples: int = 96, chunk: int = 50_000) -> np.ndarray:
    """Features for every pair (pixel p, site s): array (P*S, len(LINK_FEATURES)),
    pixel-major (row = p * S + s)."""
    P, S = len(px), len(sx)
    lam = C / freq_hz
    pp = np.repeat(np.arange(P), S)
    ss = np.tile(np.arange(S), P)
    out = np.empty((P * S, len(LINK_FEATURES)), np.float32)
    t = (np.arange(n_samples) + 0.5) / n_samples            # interior points
    for a in range(0, P * S, chunk):
        p, s = pp[a:a + chunk], ss[a:a + chunk]
        x0, y0, z0 = sx[s], sy[s], sz[s]
        x1, y1 = px[p], py[p]
        dx, dy = x1 - x0, y1 - y0
        d2 = np.hypot(dx, dy)
        d3 = np.sqrt(d2 ** 2 + (z0 - rx_h) ** 2)
        X = x0[:, None] + dx[:, None] * t
        Y = y0[:, None] + dy[:, None] * t
        H = hm.at(X, Y)
        los = z0[:, None] + (rx_h - z0)[:, None] * t
        exc = H - los
        blocked = exc > 0
        on_bld = H > 0
        n_cross = (np.diff(blocked.astype(np.int8), axis=1) == 1).sum(1) + blocked[:, 0]
        d1 = np.maximum(d2[:, None] * t, 1.0)
        dd2 = np.maximum(d2[:, None] * (1 - t), 1.0)
        nu = exc * np.sqrt(2.0 / lam * (1.0 / d1 + 1.0 / dd2))
        rx_b = hm.at(x1, y1)
        out[a:a + chunk] = np.column_stack([
            np.log10(np.maximum(d3, 1.0)), d2, z0,
            np.degrees(np.arctan2(z0 - rx_h, np.maximum(d2, 1.0))),
            blocked.mean(1), n_cross, on_bld.mean(1) * d2,
            np.maximum(exc.max(1), 0.0), knife_edge_loss_db(nu.max(1)),
            (rx_b > 0).astype(np.float32), rx_b,
            hm.at(x1, y1, hm.dens50), hm.at(x1, y1, hm.hmean50),
            hm.at(x0, y0, hm.dens100)])
    return out


def city_link_matrix(city, scene_dir, descriptors, freq_hz: float) -> np.ndarray:
    """(P*S, len(LINK_FEATURES) + D) float32 matrix for a network City: the link
    features of every (pixel, site) pair followed by the descriptors of the
    pixel's tile (`descriptors` is indexed by tile row)."""
    import json
    from pathlib import Path

    import geopandas as gpd

    from terranet.data.descriptors.scene_tiles import scene_building_parts
    from terranet.data.sionna_gen.geo import LocalFrame

    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
    parts, h = scene_building_parts(gpd.read_file(scene_dir / "building.geojson"), frame,
                                    float(meta["size_m"]))
    hm = HeightMap(parts, h, float(meta["size_m"]))
    L = link_features(hm, city.px, city.py, city.rx_h, city.sx, city.sy, city.sz, freq_hz)
    D = np.asarray(descriptors, np.float32)[city.tile]
    D = np.repeat(D, city.true_pl.shape[1], axis=0)
    return np.hstack([L, D]).astype(np.float32)
