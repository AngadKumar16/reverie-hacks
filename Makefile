.PHONY: all setup data test eda train evaluate explain app clean reproduce

PY ?= python3

all: reproduce

setup:
	$(PY) -m pip install -r requirements.txt

## Materialise the five raw CSVs and build the cached feature splits.
data:
	$(PY) -m src.data_loader
	$(PY) -m src.pipeline --force

## Leakage and correctness checks. Run these before trusting any number.
test:
	$(PY) -m pytest tests -q

eda:
	$(PY) -m src.eda

## Resumable: safe to re-run, skips work already on disk.
train:
	$(PY) -m src.train --step all

evaluate:
	$(PY) -m src.evaluate

explain:
	$(PY) -m src.explain --step shap
	$(PY) -m src.explain --step ablation

app:
	$(PY) -m streamlit run app/streamlit_app.py

## Full pipeline from a clean clone. ~15 minutes on 4 cores.
reproduce: data test eda train evaluate explain
	@echo "Done. Figures in reports/figures, metrics in reports/metrics."

clean:
	rm -rf data/processed/* models/* reports/figures/* reports/metrics/*
