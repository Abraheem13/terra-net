"""Tiny synthetic corpus with the exact on-disk layout of the real pipeline.

Used by the end-to-end smoke test so every script is exercised without the
(large) ray-traced data. Physics is deliberately simple: path loss follows a
per-block log-distance law whose exponent grows with local building height.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from terranet.data.sionna_gen.geo import LocalFrame

CITIES = {"alpha": (52.37, 4.90, 12.0), "beta": (48.86, 2.35, 20.0),
          "gamma": (41.39, 2.17, 35.0), "delta": (50.08, 14.44, 16.0)}
SIZE, STEP, N_BS = 1200.0, 10.0, 4


def write(root: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ci, (city, (lat0, lon0, h_mean)) in enumerate(CITIES.items()):
        frame = LocalFrame.from_center(lat0, lon0)
        scene = root / "data/raw/sionna_scenes" / city
        scene.mkdir(parents=True, exist_ok=True)
        polys, hs = [], []
        for cx in np.arange(-SIZE / 2 + 30, SIZE / 2 - 30, 60):
            for cy in np.arange(-SIZE / 2 + 30, SIZE / 2 - 30, 60):
                w = rng.uniform(15, 35)
                xs = np.array([-w, w, w, -w]) / 2 + cx
                ys = np.array([-w, -w, w, w]) / 2 + cy
                lat, lon = frame.to_geo(xs, ys)
                polys.append(Polygon(np.c_[lon, lat]))
                hs.append(str(round(float(rng.gamma(4, h_mean / 4)), 1)))
        gpd.GeoDataFrame({"height": hs}, geometry=polys, crs="EPSG:4326").to_file(
            scene / "building.geojson", driver="GeoJSON")
        (scene / "scene_meta.json").write_text(json.dumps(
            {"city": city, "lat0": lat0, "lon0": lon0, "size_m": SIZE}))

        g = np.arange(-SIZE / 2 + STEP / 2, SIZE / 2, STEP)
        px, py = np.meshgrid(g, g)
        px, py = px.ravel(), py.ravel()
        bs_xy = rng.uniform(-SIZE / 3, SIZE / 3, (N_BS, 2))
        gamma_field = 2.2 + h_mean / 15 + 0.3 * np.sin(px / 90) * np.cos(py / 70)
        frames = []
        for b in range(N_BS):
            d = np.sqrt((px - bs_xy[b, 0]) ** 2 + (py - bs_xy[b, 1]) ** 2 + (25 - 1.5) ** 2)
            pl = 80 + 10 * gamma_field * np.log10(d / 100) + rng.normal(0, 4, d.size)
            keep = pl <= 160
            lat, lon = frame.to_geo(px[keep], py[keep])
            blat, blon = frame.to_geo(bs_xy[b, 0], bs_xy[b, 1])
            frames.append(pd.DataFrame({
                "bs_id": f"bs{b}", "rx_lat": lat, "rx_lon": lon, "rx_h": 1.5,
                "tx_lat": float(blat), "tx_lon": float(blon), "tx_h": 25.0,
                "dist_m": d[keep], "pathloss_db": pl[keep], "los": np.nan}))
        out = root / "data/raw/sionna" / city
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(out / "measurements.parquet")
        (out / "origin.json").write_text(json.dumps({
            "source": "synthetic", "sionna_version": "none", "api": "none",
            "rt_samples": 1000, "cell_size_m": STEP, "max_pathloss_db": 160.0,
            "propagation": {"max_depth": 0}}))


def config(root: Path) -> Path:
    (root / "configs/data").mkdir(parents=True, exist_ok=True)
    (root / "configs/base.yaml").write_text(
        "paths:\n  raw: data/raw\n  processed: data/processed\n  splits: data/splits\n"
        "  outputs: outputs\n")
    (root / "configs/data/sionna_cities.yaml").write_text(
        f"size_m: {SIZE}\nbs_height_agl: 6.0\nue_height_agl: 1.5\n"
        "overture_release: synthetic\n")
    p = root / "configs/data/sionna.yaml"
    p.write_text("dataset: sionna\nscenes_dir: sionna_scenes\n"
                 f"cities: [{', '.join(CITIES)}]\nfreq_ghz: 3.5\ntile_size_m: 100.0\n"
                 "d0_m: 100.0\nmin_measurements_per_tile: 30\n")
    return p
