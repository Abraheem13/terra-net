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


# ---------------------------------------------------------------------------
# non-learned link baselines (ITU-R P.526 diffraction, 3GPP TR 38.901 UMa)
# ---------------------------------------------------------------------------
def free_space_db(d_m, freq_hz):
    return 20 * np.log10(np.maximum(d_m, 1.0)) + 20 * np.log10(freq_hz) - 147.55


def _nu(exc, d1, d2, lam):
    return exc * np.sqrt(2.0 / lam * (1.0 / np.maximum(d1, 1.0) + 1.0 / np.maximum(d2, 1.0)))


def deygout_loss_db(hm: HeightMap, px, py, rx_h, sx, sy, sz, freq_hz: float,
                    n_samples: int = 96, chunk: int = 50_000) -> np.ndarray:
    """Deygout three-edge diffraction loss (ITU-R P.526) of every (pixel, site)
    pair, pixel-major: the principal edge is the point of largest Fresnel
    parameter; the secondary edges are the largest parameters on the two
    sub-paths, counted only when the principal edge obstructs (nu > -0.78)."""
    P, S = len(px), len(sx)
    lam = C / freq_hz
    pp = np.repeat(np.arange(P), S)
    ss = np.tile(np.arange(S), P)
    out = np.empty(P * S, np.float32)
    t = (np.arange(n_samples) + 0.5) / n_samples
    for a in range(0, P * S, chunk):
        p, s = pp[a:a + chunk], ss[a:a + chunk]
        x0, y0, z0 = sx[s], sy[s], sz[s]
        x1, y1 = px[p], py[p]
        d2 = np.hypot(x1 - x0, y1 - y0)
        X = x0[:, None] + (x1 - x0)[:, None] * t
        Y = y0[:, None] + (y1 - y0)[:, None] * t
        H = hm.at(X, Y)
        los = z0[:, None] + (rx_h - z0)[:, None] * t
        exc = H - los
        dist = d2[:, None] * t
        nu = _nu(exc, dist, d2[:, None] - dist, lam)
        k = nu.argmax(1)
        rows = np.arange(len(k))
        nu_p = nu[rows, k]
        loss = knife_edge_loss_db(nu_p)
        # secondary edges on the two sub-paths (tx -> edge, edge -> rx)
        dk, hk = dist[rows, k], H[rows, k]
        zk = np.where(nu_p > -0.78, hk, los[rows, k])
        for side in (0, 1):
            if side == 0:
                m = t[None, :] < (k / n_samples)[:, None]
                z_a, d_a, z_b, d_b = z0, np.zeros_like(dk), zk, dk
            else:
                m = t[None, :] > ((k + 1) / n_samples)[:, None]
                z_a, d_a, z_b, d_b = zk, dk, np.full_like(dk, rx_h), d2
            span = np.maximum(d_b - d_a, 1.0)
            f = (dist - d_a[:, None]) / span[:, None]
            los_s = z_a[:, None] + (z_b - z_a)[:, None] * f
            exc_s = np.where(m, H - los_s, -np.inf)
            nu_s = _nu(exc_s, dist - d_a[:, None], d_b[:, None] - dist, lam)
            nu_s = np.where(m, nu_s, -np.inf)
            ks = nu_s.argmax(1)
            nu_2 = np.where(m.any(1), nu_s[rows, ks], -np.inf)
            loss = loss + np.where(nu_p > -0.78, knife_edge_loss_db(nu_2), 0.0)
        out[a:a + chunk] = loss
    return out


def uma_tr38901_db(d3, d2, h_bs, freq_hz, los, h_ut: float = 1.5) -> np.ndarray:
    """3GPP TR 38.901 UMa path loss (Table 7.4.1-1) with the LoS state given
    per link: PL_LOS with its breakpoint, PL_NLOS = max(PL_LOS, PL'_NLOS)."""
    fc = freq_hz / 1e9
    h_e = 1.0                                   # effective environment height (UMa, h_UT < 13 m)
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * freq_hz / C
    pl1 = 28.0 + 22 * np.log10(np.maximum(d3, 1.0)) + 20 * np.log10(fc)
    pl2 = (28.0 + 40 * np.log10(np.maximum(d3, 1.0)) + 20 * np.log10(fc)
           - 9 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_los = np.where(d2 <= d_bp, pl1, pl2)
    pl_nlos = 13.54 + 39.08 * np.log10(np.maximum(d3, 1.0)) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))
