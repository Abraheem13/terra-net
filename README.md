# TERRA-Net

**T**ransferable, **E**xplainable **R**adio-map **R**epresentation with **A**daptation — cross-city
transfer of interpretable path-loss parameters (γ, PL₀) for 6G digital-twin network planning.

This repository extends and rigorously benchmarks against:

> M. Ozyurt, "Descriptor-based tiling for transferable path-loss learning in 6G digital twins,"
> *Computer Communications* 257 (2026) 108620.

Our contributions over the base paper:

1. **Hybrid tile representation** — 116-dim hand-crafted descriptors fused with a learned
   CNN/GNN encoder over rasterized 3-D tile geometry, predicting interpretable (γ, PL₀).
2. **Learned transfer** — a hypernetwork / deep-metric similarity replacing the fixed Gaussian
   kernel, framed as unsupervised domain adaptation with a few-shot (k-shot) extension.
3. **Calibrated uncertainty** — heteroscedastic heads + split-conformal prediction under
   cross-city covariate shift; robustness study (descriptor corruption, missing modalities,
   temporal DT drift).
4. **BS placement with guarantees** — submodular greedy with (1−1/e) approximation on the
   learned uncertainty-aware coverage function, vs. the base paper's median heuristic.

Evaluation: **leave-one-city-out (LOCO)** over ≥6 public cities (DeepMIMO v4 city-scale,
Sionna-RT-generated European tiles, UK Ofcom drive tests, RadioMap3DSeer for comparability).
All data public; all code, seeds, and splits committed.

---

## Repository layout

```
terra-net/
├── configs/                  Hydra-style YAML configs (data / model / experiment)
├── docker/                   Reproducible CUDA container
├── src/terranet/
│   ├── utils/                Seeding, logging, geodesy, I/O
│   ├── data/
│   │   ├── download/         Dataset fetchers (DeepMIMO, RadioMapSeer, Ofcom)
│   │   ├── sionna_gen/       OSM → Mitsuba scene → Sionna RT radio-map generation
│   │   ├── descriptors/      116-dim descriptor extraction (topo / built / land / climate)
│   │   ├── tiling.py         Spherical grid construction (Alg. 2 of base paper)
│   │   ├── datasets.py       PyTorch Dataset / DataModule classes
│   │   └── splits.py         LOCO + geographic within-city splits (anti-leakage)
│   ├── models/
│   │   ├── baselines/        Log-distance, 3GPP/Okumura/ITU-R, base-paper reimpl, MLP, GP
│   │   ├── encoders/         CNN tile encoder, GNN over building graphs, fusion
│   │   ├── hypernet.py       Embedding → (γ, PL₀, σ²) hypernetwork head
│   │   ├── da/               DANN, Deep CORAL, MMD alignment
│   │   ├── uq/               Heteroscedastic loss, deep ensembles, split conformal
│   │   └── placement/        Greedy-median (base paper) + submodular placement
│   ├── training/             Trainer, losses, MAML few-shot adapter
│   └── evaluation/           Metrics, bootstrap statistics, calibration, report tables
├── scripts/                  Numbered end-to-end pipeline scripts (00_ … 07_)
├── experiments/              One YAML+shell pair per paper experiment
├── tests/                    pytest unit tests (run in CI)
└── outputs/                  checkpoints / logs / figures / tables (git-ignored)
```

## Quickstart

```bash
# 1. Environment (conda) — or use docker/Dockerfile
bash scripts/setup_env.sh
conda activate terranet

# 2. Verify install + GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
pytest tests/ -q

# 3. Phase 1 — data
bash scripts/00_download_data.sh deepmimo radiomapseer
python scripts/01_generate_sionna.py --config configs/data/sionna_gen.yaml   # optional, GPU
python scripts/02_extract_descriptors.py --config configs/data/deepmimo.yaml
python scripts/03_make_splits.py --protocol loco

# 4. Phase 2 — baselines
python scripts/04_fit_baselines.py --config configs/experiment/baselines_loco.yaml

# 5. Phase 3 — TERRA-Net
python scripts/05_train.py --config configs/experiment/terranet_loco.yaml

# 6. Phase 4 — evaluation, statistics, paper tables
python scripts/06_evaluate.py --run-dir outputs/checkpoints/terranet_loco
python scripts/07_make_tables.py --runs outputs/checkpoints/*
```

## Experiment phases and gates

| Phase | Weeks | Deliverable | Gate to proceed |
|---|---|---|---|
| 1 Data | 1–6 | All datasets tiled + descriptors extracted | Descriptor-only model reproduces log-distance-level NMSE on 1 city |
| 2 Baselines | 3–8 | Empirical + base-paper reimpl + ML baselines under LOCO | Cross-city RMSE ≈ 6–8 dB on Ofcom (literature-comparable) |
| 3 Core | 9–16 | Encoder + hypernet + DA + conformal | Significant LOCO gain over descriptor baseline AND PMNet-style transfer; PI coverage within ±3 % of nominal |
| 4 Robustness & placement | 17–22 | Corruption/missing/drift sweeps; placement comparison; ablations | Paper tables frozen |

## Reproducibility contract

- Every run: fixed seed set {0,1,2,3,4}, config snapshot, git SHA, and environment lockfile
  saved to the run directory.
- Statistics: paired bootstrap 95 % CIs across LOCO folds, Holm–Bonferroni correction.
  We do **not** report standalone Cohen's d without CIs.
- Data licenses: DeepMIMO CC BY-NC-SA 4.0 (academic use), RadioMapSeer/3DSeer CC-BY-4.0,
  Ofcom OGL, Sionna Apache-2.0. See `docs/DATA_LICENSES.md`.
