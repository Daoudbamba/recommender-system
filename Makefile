# Cibles pour Linux / macOS / CI. Sous Windows sans `make`, lancer directement
# les commandes indiquees (voir README, section Installation).
PYTHON ?= python

.PHONY: install data prepare report train train-content train-hybrid train-ranking model-report notebooks test lint all

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --no-deps -e .

data:
	$(PYTHON) scripts/download_data.py

prepare:
	$(PYTHON) scripts/prepare_data.py

report:
	$(PYTHON) scripts/build_data_report.py

train:
	$(PYTHON) scripts/train_collaborative.py

train-content:
	$(PYTHON) scripts/train_content.py

train-hybrid:
	$(PYTHON) scripts/train_hybrid.py

train-ranking:
	$(PYTHON) scripts/train_ranking.py

model-report:
	$(PYTHON) scripts/build_model_report.py

notebooks:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/01_data_exploration.ipynb notebooks/02_data_preprocessing.ipynb \
		notebooks/03_feature_engineering.ipynb notebooks/04_collaborative_filtering.ipynb \
		notebooks/05_content_based_filtering.ipynb notebooks/06_embeddings.ipynb \
		notebooks/07_hybrid_recommendation.ipynb notebooks/08_ranking_model.ipynb

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

all: data prepare report test
