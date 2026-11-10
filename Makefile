# Cibles pour Linux / macOS / CI. Sous Windows sans `make`, lancer directement
# les commandes indiquees (voir README, section Installation).
PYTHON ?= python

.PHONY: install data prepare report notebooks test lint all

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --no-deps -e .

data:
	$(PYTHON) scripts/download_data.py

prepare:
	$(PYTHON) scripts/prepare_data.py

report:
	$(PYTHON) scripts/build_data_report.py

notebooks:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/01_data_exploration.ipynb notebooks/02_data_preprocessing.ipynb

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

all: data prepare report test
