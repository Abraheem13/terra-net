"""Run Sionna RT radio-map generation over a built scene. GPU-heavy.

For each BS placement, computes a coverage/radio map; exports per-pixel path gain
as path loss (dB) with pixel coordinates, into the common measurement schema.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def run_radio_map(scene_xml: Path, bs_positions: np.ndarray, f_hz: float = 3.5e9,
                  cell_size: float = 5.0, out_npz: Path | None = None) -> dict:
    import sionna.rt as rt  # requires the [rt] environment, sionna==0.19.*

    scene = rt.load_scene(str(scene_xml))
    scene.frequency = f_hz
    results = {}
    for i, pos in enumerate(bs_positions):
        scene.remove("tx") if "tx" in scene.transmitters else None
        scene.add(rt.Transmitter(name="tx", position=pos.tolist()))
        cm = scene.coverage_map(cm_cell_size=(cell_size, cell_size), num_samples=2_000_000)
        pg = 10.0 * np.log10(np.maximum(cm.path_gain.numpy().squeeze(), 1e-30))
        results[f"bs{i}"] = {"pathloss_db": -pg, "cell_size": cell_size, "bs_pos": pos}
    if out_npz:
        np.savez_compressed(out_npz, **{k: v["pathloss_db"] for k, v in results.items()})
    return results
