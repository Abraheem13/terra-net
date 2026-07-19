#!/usr/bin/env bash
# One-shot environment setup. Requires conda (miniconda/mambaforge).
set -euo pipefail

if ! command -v conda &> /dev/null; then
  echo "conda not found. Install miniconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

conda env create -f environment.yml -n terranet || conda env update -f environment.yml -n terranet
eval "$(conda shell.bash hook)"
conda activate terranet
pip install -e .[dev]
pre-commit install

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
echo "Environment ready. Activate with: conda activate terranet"
echo "For Sionna RT data generation create a separate env:  pip install -e .[rt]"
