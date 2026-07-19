#!/usr/bin/env bash
# Fetch raw datasets. Usage: bash scripts/00_download_data.sh deepmimo radiomapseer ofcom
set -euo pipefail
for ds in "$@"; do
  case "$ds" in
    deepmimo)
      echo ">> DeepMIMO v4 city-scale (CC BY-NC-SA 4.0)"
      pip show deepmimo >/dev/null 2>&1 || pip install deepmimo
      for city in city_0_newyork city_1_losangeles city_2_chicago city_3_houston \
                  city_4_phoenix city_5_philadelphia city_6_miami city_7_sandiego \
                  city_8_dallas city_9_sanfrancisco; do
        python -m terranet.data.download.deepmimo --city "$city" --out data/raw/deepmimo || true
      done ;;
    radiomapseer)
      echo ">> RadioMap3DSeer: download the zip from IEEE DataPort (DOI 10.21227/0gtx-6v30),"
      echo "   then: python -m terranet.data.download.radiomapseer --zip <path.zip>" ;;
    ofcom)
      echo ">> Ofcom UK drive tests: verify the current landing page on Ofcom open data,"
      echo "   download city CSVs, then: python -m terranet.data.download.ofcom --csv <f> --city <c>" ;;
    *) echo "unknown dataset: $ds"; exit 1 ;;
  esac
done
