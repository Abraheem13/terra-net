#!/usr/bin/env bash
set -uo pipefail
export DRJIT_LIBLLVM_PATH=${DRJIT_LIBLLVM_PATH:-/usr/lib/llvm-18/lib/libLLVM.so.18.1}
while pgrep -f run_rt_extra.sh >/dev/null; do sleep 30; done
python scripts/01d_rt_outliers.py
python scripts/17_external_scenes.py --raytrace
echo "=== rt_ext done $(date +%T)"
