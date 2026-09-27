#!/usr/bin/env bash
# Second band (7.5 GHz, all cities) and the random-siting replicate (3.5 GHz, four cities).
set -euo pipefail
export DRJIT_LIBLLVM_PATH=${DRJIT_LIBLLVM_PATH:-/usr/lib/llvm-18/lib/libLLVM.so.18.1}
python scripts/01b_raytrace.py --config configs/data/sionna_cities_7p5.yaml
python scripts/01b_raytrace.py --config configs/data/sionna_cities_randsite.yaml
