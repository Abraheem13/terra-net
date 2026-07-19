"""UK Ofcom drive-test measurements (Open Government Licence).

~8.2M measurements, 6 frequencies (449/915/1802/2695/3602/5850 MHz), 7 cities.
Re-verify the current landing URL on Ofcom's open-data pages before running;
this module normalizes the CSVs into our schema:
  city, freq_mhz, tx_lat, tx_lon, tx_h, rx_lat, rx_lon, rx_h, pathloss_db
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCHEMA = ["city", "freq_mhz", "tx_lat", "tx_lon", "tx_h", "rx_lat", "rx_lon", "rx_h",
          "pathloss_db"]


def normalize_csv(csv_path: Path, city: str, out_dir: Path) -> Path:
    df = pd.read_csv(csv_path)
    # Column names differ per campaign file; map explicitly and fail loudly.
    raise NotImplementedError(
        f"Map raw Ofcom columns of {csv_path.name} to SCHEMA={SCHEMA} after inspecting the file."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--out", default="data/raw/ofcom")
    a = p.parse_args()
    normalize_csv(Path(a.csv), a.city, Path(a.out))
