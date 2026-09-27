# One-command reproduction.  `make reproduce` regenerates every number, table,
# figure and the PDF of the manuscript from the ray-traced corpus in data/raw/.
# Regenerating the corpus itself is `make corpus`: it downloads building
# footprints (Overture Maps, pinned release) and ray traces with Sionna RT on
# the GPU if present, else on the CPU (set DRJIT_LIBLLVM_PATH to libLLVM.so).
PY      ?= python
DATA    ?= configs/data/sionna.yaml
DATA2   ?= configs/data/sionna_7p5.yaml
CITIES  ?= configs/data/sionna_cities.yaml
CITIES2 ?= configs/data/sionna_cities_7p5.yaml
RAND    ?= configs/data/sionna_cities_randsite.yaml

.PHONY: reproduce analysis band stats paper corpus test lint clean

reproduce: analysis band stats paper

corpus:
	$(PY) scripts/01a_fetch_buildings.py --config $(CITIES)
	$(PY) scripts/01_build_scenes.py     --config $(CITIES)
	$(PY) scripts/01b_raytrace.py        --config $(CITIES)
	$(PY) scripts/01c_rt_convergence.py  --config $(CITIES)
	$(PY) scripts/01d_rt_outliers.py     --config $(CITIES)
	$(PY) scripts/01b_raytrace.py        --config $(CITIES2)
	$(PY) scripts/01b_raytrace.py        --config $(RAND)
	$(PY) scripts/17_external_scenes.py  --raytrace

analysis:
	$(PY) scripts/02_build_tiles.py    --config $(DATA)
	$(PY) scripts/03_make_splits.py    --config $(DATA)
	$(PY) scripts/04_fit_baselines.py  --config $(DATA)
	$(PY) scripts/06_train_neural.py   --config $(DATA)
	$(PY) scripts/05_kshot.py          --config $(DATA)
	$(PY) scripts/07_label_noise.py    --config $(DATA)
	$(PY) scripts/08_degeneracy.py     --config $(DATA)
	$(PY) scripts/09_network_eval.py   --config $(DATA)
	$(PY) scripts/11_link_transfer.py  --config $(DATA)
	$(PY) scripts/13_certified_coverage.py --config $(DATA)
	$(PY) scripts/12_tile_size.py      --config $(DATA)
	$(PY) scripts/14_censoring.py      --config $(DATA)
	$(PY) scripts/15_siting.py         --config $(DATA) --rand $(RAND)
	$(PY) scripts/17_external_scenes.py --config $(DATA)

# second band: the same protocol at 7.5 GHz, then cross-band transfer
band:
	$(PY) scripts/02_build_tiles.py    --config $(DATA2)
	$(PY) scripts/04_fit_baselines.py  --config $(DATA2)
	$(PY) scripts/09_network_eval.py   --config $(DATA2)
	$(PY) scripts/11_link_transfer.py  --config $(DATA2)
	$(PY) scripts/16_band.py           --config $(DATA) --band2 $(DATA2)

stats:
	$(PY) scripts/10_stats.py

paper:
	$(MAKE) -C paper

test:
	$(PY) -m pytest

lint:
	ruff check src tests scripts paper/scripts

clean:
	rm -rf outputs/predictions .pytest_cache .ruff_cache
	$(MAKE) -C paper clean
