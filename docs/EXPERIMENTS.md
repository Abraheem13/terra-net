# Experiment matrix (paper mapping)

| Table/Fig in paper | Script | Config | Runs |
|---|---|---|---|
| T1: LOCO baselines | 04_fit_baselines.py | baselines_loco.yaml | 10 folds x 7 models |
| T2: TERRA-Net vs baselines | 05_train.py + 06 + 07 | terranet_loco.yaml | 10 folds x 5 seeds |
| T3: Encoder ablation | 05 (ablations.encoder) | terranet_loco.yaml | 3 x 10 x 5 |
| T4: DA ablation | 05 (ablations.da) | terranet_loco.yaml | 4 x 10 x 5 |
| F: k-shot curves | 06 with training/maml.py | kshot in config | 5 k-values x 10 folds |
| T5: UQ calibration | 06_evaluate.py | uq block | from T2 runs |
| T6: robustness | robustness.yaml | corruption sweeps | 5 noise x 5 missing |
| T7: placement | placement modules | — | greedy vs submodular vs risk-averse |

Gate thresholds are in README. Freeze all tables before writing; any rerun
after freeze requires a changelog entry in docs/CHANGELOG_RESULTS.md.
