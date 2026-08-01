.PHONY: all setup data test eda train evaluate explain app clean reproduce \
        report notebook verify

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

## Typeset reports/report.md to PDF (needs pandoc + weasyprint).
report:
	$(PY) scripts/build_report_pdf.py

## Rebuild and execute the walkthrough notebook.
notebook:
	$(PY) scripts/build_notebook.py

app:
	$(PY) -m streamlit run app/streamlit_app.py

## 51 checks: determinism, leakage in the shipped model, and every headline
## number in the report re-read from the metrics files.
verify:
	$(PY) scripts/verify.py

## Full pipeline from a clean clone. ~15 minutes on 4 cores.
reproduce: data test eda train evaluate explain verify
	@echo "Done. Figures in reports/figures, metrics in reports/metrics."

clean:
	rm -rf data/processed/* models/* reports/figures/* reports/metrics/*
