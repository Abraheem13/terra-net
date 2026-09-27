#!/usr/bin/env bash
set -uo pipefail
until grep -q "=== rt_ext done" outputs/logs/rt_ext.log 2>/dev/null; do sleep 60; done
D2=configs/data/sionna_7p5.yaml
for s in "02_build_tiles.py --config $D2" "03_make_splits.py --config $D2" "04_fit_baselines.py --config $D2" \
         "09_network_eval.py --config $D2" "11_link_transfer.py --config $D2" \
         "16_band.py" "15_siting.py" "17_external_scenes.py" "10_stats.py"; do
  echo "=== $s $(date +%T)"; python scripts/$s || echo "FAILED $s"
done
echo "=== final done $(date +%T)"
