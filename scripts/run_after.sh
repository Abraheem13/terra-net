#!/usr/bin/env bash
set -uo pipefail
until grep -aq "=== done" outputs/logs/analysis2.log; do sleep 20; done
for s in 11_link_transfer 13_certified_coverage; do
  echo "=== $s $(date +%T)"; python scripts/$s.py --config configs/data/sionna.yaml || echo "FAILED $s"
done
echo "=== done2 $(date +%T)"
kill -CONT 600
