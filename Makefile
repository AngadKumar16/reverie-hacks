.PHONY: all setup data test eda train evaluate explain app clean reproduce \
        report notebook verify severity horizon disruption impact fairness \
        transfer cost video

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

## Severity: tiers (>15/>60/>120 min), quantile heads, conditional-on-late.
severity:
	$(PY) -m src.severity --step all

## How far ahead the model still works, using a persistence forecast.
horizon:
	$(PY) -m src.horizon

## Three-outcome disruption model over all 336,776 flights.
disruption:
	$(PY) -m src.cancellations

## Delay minutes, passenger hours, dollars and CO2 at a fixed alert budget,
## against random and no-ML baselines, with every assumption swept.
impact:
	$(PY) -m src.impact

## Who the alert budget reaches, and what evening it out would cost.
fairness:
	$(PY) -m src.fairness

## Leave-one-airport-out: does it work at an airport it has never seen?
transfer:
	$(PY) -m src.transfer

## Measured operating cost: artefact size, latency, memory, energy.
cost:
	$(PY) -m src.deploy_cost

## Typeset reports/report.md to PDF. Uses pandoc + WeasyPrint if installed,
## otherwise the `markdown` package and headless Chrome.
report:
	$(PY) scripts/build_report_pdf.py

## Rebuild and execute the walkthrough notebook.
notebook:
	$(PY) scripts/build_notebook.py

## Render docs/demo.mp4 from the figures and metrics. See docs/DEMO_VIDEO.md.
video:
	$(PY) scripts/build_demo_video.py

app:
	$(PY) -m streamlit run app/streamlit_app.py

## 119 checks: determinism, leakage in the shipped model, and every headline
## number in the report re-read from the metrics files.
verify:
	$(PY) scripts/verify.py

## Full pipeline from a clean clone. ~15 minutes on 4 cores.
reproduce: data test eda train evaluate explain severity horizon disruption \
           impact fairness transfer cost verify
	@echo "Done. Figures in reports/figures, metrics in reports/metrics."

clean:
	rm -rf data/processed/* models/* reports/figures/* reports/metrics/*
