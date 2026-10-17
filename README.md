# terra-net

Code, data and manuscript for

> **Cross-city propagation-model transfer for network planning in 6G digital
> twin networks: tile parameters versus site-aware link models**

The repository reproduces every number, table and figure of the manuscript
(`paper/main.pdf`) from the ray-traced corpora with one command.

## What is in the study

* **Corpora.** Eleven geo-referenced 3 × 3 km city scenes built from a pinned
  Overture Maps buildings release (OpenStreetMap-based), ray traced with
  Sionna RT from 20 rooftop sites per city at 3.5 GHz, the same sites at
  7.5 GHz (second band), and three replicates of four cities: sites on
  randomly chosen rooftops, sites on mid-rise rooftops (12-45 m), and the
  same geometry with ITU materials drawn at random per building. A
  convergence study traces two sites of three cities with up to 1e9 rays.
  Links more than 6 dB below free space (rare outliers of the radio-map
  estimator) are removed on loading.
* **Tile labels.** Each 100 m tile gets a log-distance pair (γ, P_L0 at 100 m),
  estimated with empirical-Bayes shrinkage whose sampling variance is the
  CR2 cluster-robust variance with the serving sites as clusters
  (`src/terranet/data/labels.py`).
* **Descriptors.** 32 built-environment features computed from exactly the
  geometry the ray tracer saw (`src/terranet/data/descriptors/scene_tiles.py`).
* **Tile transfer benchmark.** Leave-one-city-out comparison of the source
  median, 3GPP UMa, COST-231 Hata, Nadaraya–Watson kernel transfer, a Gaussian
  process with learned length scales, k-NN, ridge, CORAL, MLP, a neural
  encoder, gradient-boosted trees and importance-weighted trees. Every
  hyper-parameter is chosen by a nested inner leave-one-training-city-out
  loop (`src/terranet/experiments/common.py`), never on the held-out city;
  five seeds. Few-shot correction from random, contiguous or designed
  (facility-location) surveys, with a precision-weighted empirical-Bayes
  correction (`src/terranet/experiments/fewshot.py`).
* **Site-aware link transfer.** A hurdle model (does the link exist below
  the cut-off? how strong is it?) of every (pixel, site) link from
  obstruction and diffraction features (`src/terranet/data/link_features.py`,
  `scripts/11_link_transfer.py`), compared with per-link physics baselines
  on the same geometry (free space + knife edge, Deygout three-edge, 3GPP UMa
  with geometric LoS), and with per-site calibration from random, designed
  or stratified surveys whose strength is set by empirical Bayes
  (`src/terranet/experiments/calibration.py`).
* **Decomposition of the gap.** Receiver-tile variants (tile labels or
  transferred parameters, plus the hurdle, plus a line-of-sight dual-slope
  label) separate reach, LoS state and link geometry
  (`scripts/19_tile_variants.py`).
* **Mixed-path composition.** The tile model is also evaluated with the
  loss composed from every tile a link crosses, as in the descriptor-based
  framework it tests (`src/terranet/evaluation/mixed_path.py`).
* **Network-level evaluation.** Path loss (plain and censored at the
  cut-off), link precision and recall, reach accuracy, serving cell, SINR,
  coverage, spectral efficiency, cell-edge displacement, greedy site planning
  and the sites needed for 90 % coverage, all scored on the full ray-traced
  lattice including pixels no site reaches, on the 20-site and on a 10-site
  network (`src/terranet/evaluation/network.py`). Few-shot variants are
  scored on the unsurveyed tiles only.
* **Certified coverage.** City-level coverage certificates from the
  leave-one-city-out residuals (`scripts/13_certified_coverage.py`).
* **Robustness.** Tile size, the path-loss cut-off (Tobit refits), network
  size, three replicates (random siting, mid-rise siting, random materials),
  solver convergence, the second band including cross-band transfer, and an
  external test on the independently modelled city scenes of Sionna RT.
* **Statistics** at the city level (bootstrap CIs, exact Wilcoxon, Holm).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
make reproduce          # analysis (3.5 GHz), second band, statistics, paper assets + PDF
make test               # unit tests + an end-to-end run on a synthetic corpus
```

`make reproduce` needs no GPU and no network access; it runs from the
ray-traced corpora in `data/raw/`. The wall-clock time of every step is
recorded in `outputs/tables/runtime.csv` and reported in the experimental-setting section of the
manuscript (about one CPU-day on four cores for the nested protocol).
`paper/scripts/check_build.py` fails the LaTeX build on undefined
references, unused or missing generated macros and tables, and on any
result-like number typed by hand in the sections.

`make reproduce` runs, in order:

| Step | Script | Output (`outputs/tables/`) |
|---|---|---|
| tiles, labels, descriptors | `02_build_tiles.py` | `corpus.csv`, `fits.csv` |
| LOCO splits | `03_make_splits.py` | `splits.csv` |
| zero-shot tile transfer (nested tuning) | `04_fit_baselines.py` | `loco_metrics.csv`, `loco_hparams.csv`, `inner_bias.csv` |
| neural encoder | `06_train_neural.py` | `neural_metrics.csv` |
| few-shot correction | `05_kshot.py` | `kshot.csv` |
| label precision | `07_label_noise.py` | `label_noise.csv` |
| kernel diagnostics | `08_degeneracy.py` | `degeneracy.csv` |
| network evaluation (tile models) | `09_network_eval.py` | `network_metrics.csv`, `planning.csv` |
| site-aware link transfer, physics baselines | `11_link_transfer.py` | `network_link.csv`, `planning_link.csv`, `link_importance.csv`, `link_hparams.csv` |
| receiver-tile variants (hurdle, dual slope) | `19_tile_variants.py` | `network_variants.csv`, `planning_variants.csv`, `los_labels.csv` |
| certified coverage | `13_certified_coverage.py` | `certified.csv` |
| tile size | `12_tile_size.py` | `tile_size.csv` |
| path-loss cut-off | `14_censoring.py` | `censoring.csv` |
| siting, height and material replicates | `15_siting.py --rand <configs>` | `siting.csv` |
| external scenes (Sionna RT Munich, Etoile, Florence) | `17_external_scenes.py` | `external.csv` |
| mixed-path composition (summed as published, piecewise, path-averaged; tiles of 100, 200, 500, 1000 m) | `18_mixed_path.py [--tile-size G]` | `mixed_path[_G<size>].csv`, `mixed_path_plan[_G<size>].csv`, `mixed_path_labels[_G<size>].csv` |
| second band (7.5 GHz) | `02`, `04`, `09`, `11` with `configs/data/sionna_7p5.yaml`, then `16_band.py` | `outputs/sionna_7p5/tables/*`, `band_labels.csv`, `band_transfer.csv` |
| statistics | `10_stats.py` | `stats.csv` |
| manuscript | `paper/scripts/make_assets.py`, LaTeX | `paper/generated/*`, `paper/figures/*`, `paper/highlights.txt`, `paper/main.pdf` |

The manuscript contains no hand-typed result: every number in the text is a
macro in `paper/generated/numbers.tex`, and every table and data figure is
written by `paper/scripts/make_assets.py` from the CSVs above.

Regenerating the corpora needs network access (building footprints) and runs
the ray tracer on the GPU if present, else on the CPU:
`make corpus` (install `.[scenes,rt]`; on the CPU set `DRJIT_LIBLLVM_PATH`).
On four CPU cores one site takes about 140 s at 1e8 rays, so the main corpus
(220 sites) and the second band take about 17 hours and each replicate (80
sites) about 3 hours. The convergence study (`01c_rt_convergence.py`) is the
only part that is expensive at 1e9 rays.

The ray-traced corpora (`data/raw/`) are versioned with the scene
definitions (`data/raw/sionna_scenes/<city>/scene_meta.json`); an archived copy with a DOI
will be deposited on acceptance.

## Layout

```
configs/            data configuration (cities, bands, tile size, reference distance)
src/terranet/
  data/             tiling, labels, descriptors, link features, scene and ray-tracing code
  experiments/      shared LOCO protocol and transfer operators
  evaluation/       network evaluation, metrics and statistics
  models/           neural encoder/head, heteroscedastic loss, placement
scripts/            numbered pipeline (00-19)
paper/              manuscript sources, asset generator, generated assets
tests/              unit tests and the synthetic end-to-end test
```

## Licence

Code: MIT. Building geometry derives from OpenStreetMap via Overture Maps
(ODbL); see `DATA.md`.
