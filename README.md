# terra-net

Code, data and manuscript for

> **Cross-city path-loss transfer for digital-twin network planning: why tile
> parameters fall short and how site-aware link models close the gap**

The repository reproduces every number, table and figure of the manuscript
(`paper/main.pdf`) from the ray-traced corpora with one command.

## What is in the study

* **Corpora.** Eleven geo-referenced 3 × 3 km city scenes built from a pinned
  Overture Maps buildings release (OpenStreetMap-based), ray traced with
  Sionna RT from 20 rooftop sites per city at 3.5 GHz, the same sites at
  7.5 GHz (second band), and four cities re-traced with sites on randomly
  chosen rooftops (siting replicate). Links more than 6 dB below free space
  (rare outliers of the radio-map estimator) are removed on loading.
* **Tile labels.** Each 100 m tile gets a log-distance pair (γ, P_L0 at 100 m),
  estimated with empirical-Bayes shrinkage whose sampling variance is the
  CR2 cluster-robust variance with the serving sites as clusters
  (`src/terranet/data/labels.py`).
* **Descriptors.** 32 built-environment features computed from exactly the
  geometry the ray tracer saw (`src/terranet/data/descriptors/scene_tiles.py`).
* **Tile transfer benchmark.** Leave-one-city-out comparison of the source
  median, 3GPP UMa, COST-231 Hata, Nadaraya–Watson kernel transfer, a Gaussian
  process with learned length scales, k-NN, ridge, CORAL, MLP, a neural
  encoder, gradient-boosted trees and importance-weighted trees, with one
  tuning protocol and five seeds; few-shot correction from random,
  contiguous or designed (facility-location) surveys.
* **Site-aware link transfer.** A gradient-boosted model of every
  (pixel, site) link from obstruction and diffraction features
  (`src/terranet/data/link_features.py`, `scripts/11_link_transfer.py`), with
  per-site calibration from random or designed MDT-style surveys.
* **Network-level evaluation.** Path loss, serving cell, SINR, coverage,
  spectral efficiency, cell-edge displacement, greedy site planning and the
  sites needed for 90 % coverage, all scored on the ray-traced truth, on the
  20-site and on a 10-site network (`src/terranet/evaluation/network.py`).
* **Certified coverage.** City-level coverage certificates from the
  leave-one-city-out residuals (`scripts/13_certified_coverage.py`).
* **Robustness.** Tile size, the path-loss cut-off (Tobit refits), the siting
  policy and the second band, including cross-band transfer.
* **Statistics** at the city level (bootstrap CIs, exact Wilcoxon, Holm).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
make reproduce          # analysis (3.5 GHz), second band, statistics, paper assets + PDF
make test               # unit tests + an end-to-end run on a synthetic corpus
```

`make reproduce` runs, in order:

| Step | Script | Output (`outputs/tables/`) |
|---|---|---|
| tiles, labels, descriptors | `02_build_tiles.py` | `corpus.csv`, `fits.csv` |
| LOCO splits | `03_make_splits.py` | `splits.csv` |
| zero-shot tile transfer | `04_fit_baselines.py` | `loco_metrics.csv`, `loco_hparams.csv` |
| neural encoder | `06_train_neural.py` | `neural_metrics.csv` |
| few-shot correction | `05_kshot.py` | `kshot.csv` |
| label precision | `07_label_noise.py` | `label_noise.csv` |
| kernel diagnostics | `08_degeneracy.py` | `degeneracy.csv` |
| network evaluation (tile models) | `09_network_eval.py` | `network_metrics.csv`, `planning.csv` |
| site-aware link transfer | `11_link_transfer.py` | `network_link.csv`, `planning_link.csv`, `link_importance.csv` |
| certified coverage | `13_certified_coverage.py` | `certified.csv` |
| tile size | `12_tile_size.py` | `tile_size.csv` |
| path-loss cut-off | `14_censoring.py` | `censoring.csv` |
| siting policy | `15_siting.py` | `siting.csv` |
| second band (7.5 GHz) | `02`, `04`, `09`, `11` with `configs/data/sionna_7p5.yaml`, then `16_band.py` | `outputs/sionna_7p5/tables/*`, `band_labels.csv`, `band_transfer.csv` |
| statistics | `10_stats.py` | `stats.csv` |
| manuscript | `paper/scripts/make_assets.py`, LaTeX | `paper/generated/*`, `paper/figures/*`, `paper/highlights.txt`, `paper/main.pdf` |

The manuscript contains no hand-typed result: every number in the text is a
macro in `paper/generated/numbers.tex`, and every table and data figure is
written by `paper/scripts/make_assets.py` from the CSVs above.

Regenerating the corpora needs network access (building footprints) and runs
the ray tracer on the GPU if present, else on the CPU:
`make corpus` (install `.[scenes,rt]`; on the CPU set `DRJIT_LIBLLVM_PATH`).

## Layout

```
configs/            data configuration (cities, bands, tile size, reference distance)
src/terranet/
  data/             tiling, labels, descriptors, link features, scene and ray-tracing code
  experiments/      shared LOCO protocol and transfer operators
  evaluation/       network evaluation, metrics and statistics
  models/           neural encoder/head, heteroscedastic loss, placement
scripts/            numbered pipeline (01-16)
paper/              manuscript sources, asset generator, generated assets
tests/              unit tests and the synthetic end-to-end test
```

## Licence

Code: MIT. Building geometry derives from OpenStreetMap via Overture Maps
(ODbL); see `DATA.md`.
