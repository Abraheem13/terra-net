"""RadioMapSeer / RadioMap3DSeer fetcher (CC-BY-4.0).

Data: https://radiomapseer.github.io  and IEEE DataPort DOI 10.21227/0gtx-6v30.
Downloads require a browser-authenticated grab for DataPort; we accept a local
zip path and unpack into data/raw/radiomapseer.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def unpack(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print(f"Unpacked {zip_path} -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--out", default="data/raw/radiomapseer")
    a = p.parse_args()
    unpack(Path(a.zip), Path(a.out))
