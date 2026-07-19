.PHONY: env test lint format data baselines train evaluate tables clean

env:
	bash scripts/setup_env.sh

test:
	pytest tests/ -q

lint:
	ruff check src tests scripts
	mypy src/terranet

format:
	ruff format src tests scripts

data:
	bash scripts/00_download_data.sh deepmimo radiomapseer
	python scripts/02_extract_descriptors.py --config configs/data/deepmimo.yaml
	python scripts/03_make_splits.py --protocol loco

baselines:
	python scripts/04_fit_baselines.py --config configs/experiment/baselines_loco.yaml

train:
	python scripts/05_train.py --config configs/experiment/terranet_loco.yaml

evaluate:
	python scripts/06_evaluate.py --run-dir outputs/checkpoints/terranet_loco

tables:
	python scripts/07_make_tables.py --runs "outputs/checkpoints/*"

clean:
	rm -rf outputs/logs/* .pytest_cache .ruff_cache
