#!/usr/bin/env python
"""Fetch building footprints for every scene from Overture Maps (buildings theme).

Overture's building layer is a versioned, openly licensed conflation of
OpenStreetMap with other open building datasets. A fixed release makes the
geometry exactly reproducible, which a live OpenStreetMap query is not.

Access: the public S3 bucket over HTTPS. The per-row-group bounding boxes of
every file are indexed once (cached), then only the row groups intersecting a
scene are read.

Output per city: data/raw/sionna_scenes/<city>/building.geojson with columns
  height            metres, from the source record (NaN if absent)
  building:levels   number of floors (NaN if absent)
  footprint_source  dataset that contributed the footprint geometry
and data/raw/sionna_scenes/<city>/building_source.json (release, counts).
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import shapely
from omegaconf import OmegaConf

from terranet.data.sionna_gen.geo import LocalFrame

BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"
PREFIX = "release/{release}/theme=buildings/type=building/"
COLUMNS = ["id", "sources", "height", "num_floors", "is_underground", "geometry", "bbox"]


def list_files(release: str) -> list[str]:
    r = requests.get(BUCKET, params={"list-type": "2", "prefix": PREFIX.format(release=release)},
                     timeout=60)
    r.raise_for_status()
    assert "<IsTruncated>false" in r.text, "listing truncated"
    return [f"{BUCKET}/{k}" for k in re.findall(r"<Key>([^<]+\.parquet)</Key>", r.text)]


def _fs():
    return fsspec.filesystem("https", client_kwargs={"trust_env": True})


def footer_index(url: str) -> list[dict]:
    with _fs().open(url, block_size=2 ** 20) as f:
        md = pq.read_metadata(f)
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    ix = {k: names.index(f"bbox.{k}") for k in ("xmin", "xmax", "ymin", "ymax")}
    out = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        out.append({"url": url, "rg": g,
                    "xmin": rg.column(ix["xmin"]).statistics.min,
                    "xmax": rg.column(ix["xmax"]).statistics.max,
                    "ymin": rg.column(ix["ymin"]).statistics.min,
                    "ymax": rg.column(ix["ymax"]).statistics.max})
    return out


def build_index(release: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    files = list_files(release)
    with ThreadPoolExecutor(16) as ex:
        rows = [r for part in ex.map(footer_index, files) for r in part]
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def read_bbox(index: pd.DataFrame, w, s, e, n) -> pd.DataFrame:
    hit = index[(index.xmax >= w) & (index.xmin <= e) & (index.ymax >= s) & (index.ymin <= n)]
    frames = []
    for url, g in hit.groupby("url"):
        with _fs().open(url, block_size=8 * 2 ** 20) as f:
            t = pq.ParquetFile(f).read_row_groups(sorted(g.rg.tolist()), columns=COLUMNS)
        d = t.to_pandas()
        bb = pd.json_normalize(d.pop("bbox"))
        keep = ((bb.xmax >= w) & (bb.xmin <= e) & (bb.ymax >= s) & (bb.ymin <= n)).to_numpy()
        frames.append(d[keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def footprint_source(sources) -> str:
    for s in sources if sources is not None else []:
        if s.get("property") in ("", None):
            return str(s.get("dataset"))
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna_cities.yaml")
    ap.add_argument("--city", default=None)
    ap.add_argument("--out", default="data/raw/sionna_scenes")
    ap.add_argument("--pad-m", type=float, default=50.0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    release = str(cfg.overture_release)
    index = build_index(release, Path("data/raw/overture") / f"index_{release}.parquet")
    for city, (lat, lon) in cfg.cities.items():
        if args.city and city != args.city:
            continue
        out = Path(args.out) / city
        if (out / "building.geojson").exists():
            print(f"[skip] {city}")
            continue
        frame = LocalFrame.from_center(float(lat), float(lon))
        w, s, e, n = frame.bbox_deg(float(cfg.size_m), pad_m=args.pad_m)
        d = read_bbox(index, w, s, e, n)
        d = d[~d.is_underground.fillna(False).astype(bool)]
        geom = shapely.from_wkb(d.geometry.to_numpy())
        g = gpd.GeoDataFrame({
            "id": d.id.to_numpy(),
            "height": d.height.astype(float).to_numpy(),
            "building:levels": d.num_floors.astype(float).to_numpy(),
            "footprint_source": [footprint_source(x) for x in d.sources],
        }, geometry=geom, crs="EPSG:4326")
        g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        out.mkdir(parents=True, exist_ok=True)
        g.to_file(out / "building.geojson", driver="GeoJSON")
        src = g.footprint_source.value_counts().to_dict()
        (out / "building_source.json").write_text(json.dumps({
            "provider": "Overture Maps Foundation", "theme": "buildings",
            "release": release, "bbox_wsen": [w, s, e, n], "n_buildings": int(len(g)),
            "footprint_sources": {k: int(v) for k, v in src.items()},
            "frac_with_height": float(np.isfinite(g.height).mean()),
            "frac_with_levels": float(np.isfinite(g["building:levels"]).mean()),
        }, indent=2))
        print(f"[ok] {city}: {len(g)} buildings, sources {src}")


if __name__ == "__main__":
    main()
