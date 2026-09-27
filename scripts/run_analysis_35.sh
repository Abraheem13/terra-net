#!/usr/bin/env bash
set -euo pipefail
for s in 02_build_tiles 03_make_splits 04_fit_baselines 06_train_neural 05_kshot 07_label_noise 08_degeneracy 09_network_eval 11_link_transfer 13_certified_coverage 14_censoring 12_tile_size; do
  echo "=== $s $(date +%T)"; python scripts/$s.py --config configs/data/sionna.yaml
done
echo "=== done $(date +%T)"
