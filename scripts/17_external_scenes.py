#!/usr/bin/env python
"""Cross-dataset test on the city scenes distributed with Sionna RT.

The Munich, Etoile and Florence scenes of Sionna RT were modelled
independently of this corpus (detailed meshes, several ITU materials: brick,
concrete, marble, metal, wood). They are used as an external test set; no
model is trained on them. (San Francisco has terrain and would need a
terrain-following measurement surface; it is not used.)

Phase 1 (--raytrace): for every scene
  * building heights on a 2 m raster by vertical ray casting of the mesh;
    footprints are the connected components of the raster above 1 m, each
    with its median height (the input of the descriptors and link features);
  * SITES rooftop sites by the rule of the main corpus (tallest local maxima
    of a 25 m grid, minimum separation MIN_SEP m, 6 m mast);
  * radio maps with the settings of configs/data/sionna_cities.yaml on the
    scene rectangle at 10 m, stored in the corpus format under
    data/raw/sionna_ext/<scene>/ (measurements.parquet, origin.json,
    building.geojson, scene_meta.json).
Phase 2 (default): tile labels and descriptors at 100 m, then the network
evaluation of the tile labels (oracle), the source median and GBDT tile
transfer fitted on all eleven cities, and the site-aware link model (the
mean of the eleven leave-one-city-out hurdle models, seed 0), zero-shot and
with the per-site calibration of 11_link_transfer.py from k surveyed tiles
(random: ten draws; "d": the survey design).

Output: outputs/tables/external.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from terranet.data.descriptors.scene_tiles import (
    DESCRIPTOR_COLUMNS,
    scene_building_parts,
    tile_descriptors,
)
from terranet.data.labels import fit_tiles, link_distance_m, load_links, log_distance
from terranet.data.sionna_gen.geo import LocalFrame
from terranet.data.tiling import grid_frame, point_tile_index, region_from_points
from terranet.evaluation.network import City
from terranet.experiments.calibration import calibrate, design_survey
from terranet.experiments.common import TARGETS, MaxAbs, fit_gbdt, load_cfg, stack
from terranet.utils.logging import get_logger

log = get_logger("external")
SCENES = {"munich": (48.1374, 11.5755), "etoile": (48.8738, 2.2950),
          "florence": (43.7731, 11.2560)}
SITES, MIN_SEP, MAST, RASTER = 8, 150.0, 6.0, 2.0
SURVEY_KS, N_DRAWS = (10, 25), 10
DATASET = "sionna_ext"


def height_raster(scene, bb, res):
    """Top surface height (m above the z = 0 ground) on a regular grid."""
    import drjit as dr
    import mitsuba as mi
    xs = np.arange(bb.min.x + res / 2, bb.max.x, res)
    ys = np.arange(bb.min.y + res / 2, bb.max.y, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    z0 = float(bb.max.z) + 10.0
    o = mi.Point3f(mi.Float(X.ravel()), mi.Float(Y.ravel()), mi.Float(np.full(X.size, z0)))
    d = mi.Vector3f(0.0, 0.0, -1.0)
    si = scene.mi_scene.ray_intersect(mi.Ray3f(o, d))
    t = np.asarray(dr.select(si.is_valid(), si.t, z0))
    return xs, ys, np.maximum(z0 - t, 0.0).reshape(X.shape)


def footprints(xs, ys, H, min_h=1.0):
    """Connected components of the raster above min_h -> polygons + median height."""
    from rasterio import features
    from rasterio.transform import from_origin
    from scipy import ndimage
    from shapely.geometry import shape
    mask = H > min_h
    lab, n = ndimage.label(mask)
    res = xs[1] - xs[0]
    # raster rows = y descending for rasterio
    lab_img = lab.T[::-1].astype(np.int32)
    tr = from_origin(xs[0] - res / 2, ys[-1] + res / 2, res, res)
    med = ndimage.median(H, lab, index=np.arange(1, n + 1))
    parts, heights = [], []
    for geom, val in features.shapes(lab_img, mask=lab_img > 0, transform=tr):
        p = shape(geom)
        if p.area >= 4.0:
            parts.append(p)
            heights.append(float(med[int(val) - 1]))
    return parts, np.asarray(heights)


def rooftop_sites(xs, ys, H, n, min_sep, mast):
    cell = 25.0
    ix = ((xs - xs[0]) // cell).astype(int)
    iy = ((ys - ys[0]) // cell).astype(int)
    cand = []
    for bx in np.unique(ix):
        for by in np.unique(iy):
            sub = H[np.ix_(ix == bx, iy == by)]
            if sub.size == 0:
                continue
            k = np.unravel_index(np.argmax(sub), sub.shape)
            h = sub[k]
            if h > 6.0:
                cand.append((xs[ix == bx][k[0]], ys[iy == by][k[1]], h))
    cand.sort(key=lambda c: -c[2])
    chosen = []
    for c in cand:
        if len(chosen) == n:
            break
        if all(np.hypot(c[0] - q[0], c[1] - q[1]) >= min_sep for q in chosen):
            chosen.append(c)
    return np.array([[x, y, h + mast] for x, y, h in chosen])


def raytrace(gen, raw):
    from terranet.data.sionna_gen.rt_runner import select_variant
    variant = select_variant()
    import geopandas as gpd
    import mitsuba as mi
    import sionna.rt as rt
    from shapely.geometry import Polygon
    for name, (lat0, lon0) in SCENES.items():
        out = raw / DATASET / name
        if (out / "measurements.parquet").exists():
            log.info(f"[skip] {name}")
            continue
        scene = rt.load_scene(getattr(rt.scene, name))
        scene.frequency = float(gen.freq_ghz) * 1e9
        for a in ("tx_array", "rx_array"):
            setattr(scene, a, rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                             polarization="V"))
        bb = scene.mi_scene.bbox()
        xs, ys, H = height_raster(scene, bb, RASTER)
        parts, heights = footprints(xs, ys, H)
        frame = LocalFrame.from_center(lat0, lon0)
        out.mkdir(parents=True, exist_ok=True)
        # the local frame of the corpus is centred on the scene: shift by its centre
        cx, cy = float(bb.min.x + bb.max.x) / 2, float(bb.min.y + bb.max.y) / 2

        def to_geo_poly(p):
            x, y = np.asarray(p.exterior.coords).T
            lat, lon = frame.to_geo(x - cx, y - cy)
            return Polygon(np.c_[lon, lat])
        gpd.GeoDataFrame({"height": heights}, geometry=[to_geo_poly(p) for p in parts],
                         crs="EPSG:4326").to_file(out / "building.geojson", driver="GeoJSON")
        cs = float(gen.cell_size_m)
        sx = float(np.floor(float(bb.max.x - bb.min.x) / cs) * cs)
        sy = float(np.floor(float(bb.max.y - bb.min.y) / cs) * cs)
        sites = rooftop_sites(xs, ys, H, SITES, MIN_SEP, MAST)
        solver = rt.RadioMapSolver()
        prop = dict(max_depth=int(gen.max_depth), los=True, specular_reflection=True,
                    diffuse_reflection=True, refraction=True, diffraction=True,
                    edge_diffraction=True)
        frames, t0 = [], time.time()
        for i, pos in enumerate(sites):
            for tx in list(scene.transmitters):
                scene.remove(tx)
            scene.add(rt.Transmitter(name=f"bs{i}", position=[float(v) for v in pos]))
            rm = solver(scene, center=mi.Point3f(cx, cy, float(gen.ue_height_agl)),
                        orientation=mi.Point3f(0.0, 0.0, 0.0), size=mi.Point2f(sx, sy),
                        cell_size=mi.Point2f(float(gen.cell_size_m), float(gen.cell_size_m)),
                        samples_per_tx=int(gen.rt_samples), seed=i, **prop)
            pg = np.asarray(rm.path_gain)[0].reshape(-1)
            cc = np.asarray(rm.cell_centers).reshape(-1, 3)
            with np.errstate(divide="ignore"):
                pl = -10.0 * np.log10(pg)
            ok = np.isfinite(pl) & (pl <= float(gen.max_pathloss_db))
            lat, lon = frame.to_geo(cc[ok, 0] - cx, cc[ok, 1] - cy)
            blat, blon = frame.to_geo(pos[0] - cx, pos[1] - cy)
            dist = np.linalg.norm(cc[ok] - pos, axis=1)
            frames.append(pd.DataFrame({
                "bs_id": f"bs{i:02d}", "rx_lat": lat, "rx_lon": lon, "rx_h": cc[ok, 2],
                "tx_lat": float(blat), "tx_lon": float(blon), "tx_h": float(pos[2]),
                "dist_m": dist, "pathloss_db": pl[ok]}))
            log.info(f"{name} bs{i:02d}: {ok.sum()} cells")
        pd.concat(frames, ignore_index=True).to_parquet(out / "measurements.parquet")
        size = float(max(sx, sy))
        meta = {"lat0": lat0, "lon0": lon0, "size_m": size, "center_local": [cx, cy],
                "extent_m": [sx, sy], "source": f"sionna.rt.scene.{name}"}
        (out / "scene_meta.json").write_text(json.dumps(meta, indent=2))
        run = {"source": "sionna_rt", "scene": f"sionna.rt.scene.{name}",
               "sionna_version": rt.__version__, "mitsuba_variant": variant,
               "freq_ghz": float(gen.freq_ghz), "cell_size_m": float(gen.cell_size_m),
               "rt_samples": int(gen.rt_samples), "n_bs": len(sites), "propagation": prop,
               "max_pathloss_db": float(gen.max_pathloss_db), "min_sep_m": MIN_SEP,
               "materials": sorted({o.radio_material.name for o in scene.objects.values()}),
               "n_footprints": len(parts), "wall_clock_s": round(time.time() - t0, 1)}
        (out / "origin.json").write_text(json.dumps(run, indent=2))
        log.info(f"[ok] {name}: {len(sites)} sites, {len(parts)} footprints")


def evaluate(cfg, base):
    import geopandas as gpd
    import lightgbm as lgb

    from terranet.data.link_features import city_link_matrix
    from terranet.experiments.common import fit_ridge
    raw = Path(base.paths.raw)
    out_dir = Path(base.paths.outputs)
    # tile transfer fitted on all eleven cities (validation: a fixed held-out city)
    splits = json.loads(Path("data/splits/loco.json").read_text())
    val = splits[0]["val_cities"][0]
    tr = stack(base, cfg, [c for c in cfg.cities if c != val])
    va = stack(base, cfg, [val])
    norm = MaxAbs().fit(tr[DESCRIPTOR_COLUMNS].to_numpy(float))
    Xtr, Ytr = norm(tr[DESCRIPTOR_COLUMNS].to_numpy(float)), tr[TARGETS].to_numpy(float)
    Xva, Yva = norm(va[DESCRIPTOR_COLUMNS].to_numpy(float)), va[TARGETS].to_numpy(float)
    gbdt = fit_gbdt(Xtr, Ytr, Xva, Yva, seed=0)
    ridge, _ = fit_ridge(Xtr, Ytr, Xva, Yva)
    med = np.median(np.r_[Ytr, Yva], 0)
    rows = []
    for si, name in enumerate(SCENES):
        d = raw / DATASET / name
        meas, n_qc = load_links(d / "measurements.parquet", cfg)
        meta = json.loads((d / "scene_meta.json").read_text())
        frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
        grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()),
                          float(cfg.tile_size_m))
        idx = point_tile_index(grid, meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy())
        x = log_distance(link_distance_m(meas), float(cfg.d0_m))
        f = fit_tiles(idx, x, meas.pathloss_db.to_numpy(float), meas.bs_id.to_numpy(),
                      len(grid), min_count=int(cfg.min_measurements_per_tile))
        bld = gpd.read_file(d / "building.geojson")
        parts, heights = scene_building_parts(bld, frame, float(meta["size_m"]))
        D = tile_descriptors(grid, parts, heights, frame)
        tiles = pd.concat([grid.reset_index(drop=True), f.frame(), D], axis=1)
        city = City(meas, grid, tiles, float(cfg.d0_m), frame)
        ok = np.isfinite(tiles.gamma.to_numpy())
        Xt = norm(tiles[DESCRIPTOR_COLUMNS].to_numpy(float))
        preds = {"oracle": tiles[TARGETS].to_numpy(float),
                 "median": np.tile(med, (len(tiles), 1)),
                 "ridge": ridge.predict(Xt), "gbdt": gbdt.predict(Xt)}
        info = {"scene": name, "links": len(meas), "qc_removed": n_qc, "tiles": int(ok.sum()),
                "sites": int(meas.bs_id.nunique()), "footprints": len(bld),
                "height_median_m": float(np.median(heights))}
        for op, P in preds.items():
            M = city.predicted_pl(P[:, 0], P[:, 1])
            r = {**info, "operator": op, "draw": 0, **city.evaluate(M)}
            r["regret5"] = next(q["regret"] for q in city.plan(M) if q["K"] == 5)
            if op != "oracle":
                e = P[ok] - tiles[TARGETS].to_numpy(float)[ok]
                r["gamma_rmse"] = float(np.sqrt((e[:, 0] ** 2).mean()))
            rows.append(r)
        X = city_link_matrix(city, d, tiles[DESCRIPTOR_COLUMNS].to_numpy(np.float32),
                             float(cfg.freq_ghz) * 1e9)
        R = np.zeros(city.true_pl.shape)
        Pk = np.zeros(city.true_pl.shape)
        for c in cfg.cities:
            R += lgb.Booster(model_file=str(out_dir / "models" / f"link_{c}_0.txt")) \
                .predict(X).reshape(R.shape)
            Pk += lgb.Booster(model_file=str(out_dir / "models" / f"linkclf_{c}_0.txt")) \
                .predict(X).reshape(R.shape)
        R /= len(cfg.cities)
        Pk /= len(cfg.cities)
        keep = Pk >= 0.5
        M = np.where(keep, R, np.inf)

        def score(op, M, R, draw=0):
            r = {**info, "operator": op, "draw": draw, **city.evaluate(M, pl_for_error=R)}
            r["regret5"] = next(q["regret"] for q in city.plan(M) if q["K"] == 5)
            rows.append(r)
        score("link", M, R)
        # the same per-site calibration from k surveyed tiles as 11_link_transfer.py
        tile_ids = np.flatnonzero(ok)
        for k in SURVEY_KS:
            for draw in range(N_DRAWS):
                pick = np.random.default_rng([si, k, draw]).choice(len(tile_ids), size=k,
                                                                   replace=False)
                score(f"link+k{k}", *calibrate(city, R, keep, tile_ids[pick]), draw=draw)
            pick = design_survey(city.tile, tile_ids, keep, k)
            score(f"link+k{k}d", *calibrate(city, R, keep, tile_ids[pick]))
        log.info(f"{name}: " + " ".join(f"{q['operator']}={q['pl_rmse_db']:.2f}"
                                         for q in rows if q["scene"] == name))
    o = out_dir / "tables" / "external.csv"
    pd.DataFrame(rows).to_csv(o, index=False)
    log.info(f"wrote {o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--gen", default="configs/data/sionna_cities.yaml")
    ap.add_argument("--raytrace", action="store_true")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    if args.raytrace:
        raytrace(OmegaConf.load(args.gen), Path(base.paths.raw))
    else:
        evaluate(cfg, base)


if __name__ == "__main__":
    main()
