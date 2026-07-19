"""DeepMIMO v4 city-scale fetcher.

License: CC BY-NC-SA 4.0 (academic use). Scenarios are downloaded through the
official `deepmimo` package / deepmimo.net; this module wraps generation of
per-link path loss + UE/BS coordinates into our parquet layout.

Usage (in the terranet env, after `pip install deepmimo`):
    python -m terranet.data.download.deepmimo --city city_0_newyork --out data/raw/deepmimo
"""
from __future__ import annotations

import argparse
from pathlib import Path

CITIES_V4 = [
    "city_0_newyork", "city_1_losangeles", "city_2_chicago", "city_3_houston",
    "city_4_phoenix", "city_5_philadelphia", "city_6_miami", "city_7_sandiego",
    "city_8_dallas", "city_9_sanfrancisco",
]


def fetch(city: str, out_dir: Path) -> Path:
    try:
        import deepmimo as dm  # type: ignore
    except ImportError as e:
        raise SystemExit("pip install deepmimo  (see deepmimo.net for scenario files)") from e
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = dm.load(city)  # downloads scenario on first use per DeepMIMO docs
    # Extract per-user: position, pathloss (dB), LoS flag; and BS positions.
    # DeepMIMO API surface changes between versions — keep this thin and assert.
    raise NotImplementedError(
        "Map dataset fields to parquet here after pinning the deepmimo version; "
        "required columns: ue_lat, ue_lon, bs_lat, bs_lon, pathloss_db, los."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, choices=CITIES_V4)
    p.add_argument("--out", default="data/raw/deepmimo")
    a = p.parse_args()
    fetch(a.city, Path(a.out))
