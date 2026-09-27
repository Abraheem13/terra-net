#!/usr/bin/env bash
set -uo pipefail
for s in 09_network_eval 11_link_transfer 13_certified_coverage; do
  echo "=== $s $(date +%T)"; python scripts/$s.py --config configs/data/sionna.yaml || echo "FAILED $s"
done
python scripts/10_stats.py > /dev/null && echo "=== stats ok"
echo "=== done3 $(date +%T)"
kill -CONT 600
