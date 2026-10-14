#!/usr/bin/env python
"""Run the complete analysis in order and record the wall-clock time of every
step (outputs/tables/runtime.csv), so that the manuscript can state the cost
of one reproduction. `make reproduce` calls this script; a single step can be
rerun with --only <script name prefix>.
"""
import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

DATA, DATA2 = "configs/data/sionna.yaml", "configs/data/sionna_7p5.yaml"
RAND = ["configs/data/sionna_cities_randsite.yaml", "configs/data/sionna_cities_capped.yaml",
        "configs/data/sionna_cities_materials.yaml"]
STEPS = [
    # analysis at 3.5 GHz
    ("02_build_tiles.py", ["--config", DATA]),
    ("03_make_splits.py", ["--config", DATA]),
    ("04_fit_baselines.py", ["--config", DATA]),
    ("06_train_neural.py", ["--config", DATA]),
    ("05_kshot.py", ["--config", DATA]),
    ("07_label_noise.py", ["--config", DATA]),
    ("08_degeneracy.py", ["--config", DATA]),
    ("11_link_transfer.py", ["--config", DATA]),
    ("09_network_eval.py", ["--config", DATA]),
    ("19_tile_variants.py", ["--config", DATA]),
    ("13_certified_coverage.py", ["--config", DATA]),
    ("12_tile_size.py", ["--config", DATA]),
    ("14_censoring.py", ["--config", DATA]),
    ("15_siting.py", ["--config", DATA, "--rand", *RAND]),
    ("17_external_scenes.py", ["--config", DATA]),
    ("18_mixed_path.py", ["--config", DATA]),
    ("18_mixed_path.py", ["--config", DATA, "--tile-size", "200"]),
    ("18_mixed_path.py", ["--config", DATA, "--tile-size", "500"]),
    ("18_mixed_path.py", ["--config", DATA, "--tile-size", "1000"]),
    # second band
    ("02_build_tiles.py", ["--config", DATA2]),
    ("03_make_splits.py", ["--config", DATA2]),
    ("04_fit_baselines.py", ["--config", DATA2]),
    ("11_link_transfer.py", ["--config", DATA2]),
    ("09_network_eval.py", ["--config", DATA2]),
    ("16_band.py", ["--config", DATA, "--band2", DATA2]),
    # inference
    ("10_stats.py", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only the steps whose script starts with this")
    ap.add_argument("--from", dest="start", default=None, help="start at this script")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("TERRANET_JOBS", 1)),
                    help="held-out cities evaluated in parallel where a step supports it")
    args = ap.parse_args()
    os.environ["TERRANET_JOBS"] = str(args.jobs)
    out = Path("outputs/tables/runtime.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists() and (args.only or args.start):
        rows = list(csv.DictReader(out.open()))
    started = args.start is None
    for script, extra in STEPS:
        if not started:
            started = script.startswith(args.start)
            if not started:
                continue
        if args.only and not script.startswith(args.only):
            continue
        t = time.time()
        print(f"==> {script} {' '.join(extra)}", flush=True)
        r = subprocess.run([sys.executable, f"scripts/{script}", *extra])
        if r.returncode:
            sys.exit(f"{script} failed with exit code {r.returncode}")
        rows = [x for x in rows if not (x["script"] == script and x["args"] == " ".join(extra))]
        rows.append({"script": script, "args": " ".join(extra),
                     "seconds": f"{time.time() - t:.0f}", "machine": platform.machine(),
                     "cpus": str(os.cpu_count()), "jobs": str(args.jobs)})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["script", "args", "seconds", "machine", "cpus",
                                              "jobs"])
            w.writeheader()
            w.writerows(rows)
    print(f"total {sum(float(r['seconds']) for r in rows) / 3600:.1f} h")


if __name__ == "__main__":
    main()
