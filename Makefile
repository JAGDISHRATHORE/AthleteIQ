.PHONY: install data build train sample figures evals notebooks test lint app all

install:      ## install runtime + dev dependencies
	pip install -r requirements-dev.txt
data:         ## download the public dataset (~330 MB)
	python scripts/run_pipeline.py download
build:        ## quality checks, metrics, features, clusters
	python scripts/run_pipeline.py build
train:        ## baselines, models, intervals, SHAP -> models/metrics.json
	python scripts/run_pipeline.py train
sample:       ## refresh the bundled 12-athlete sample
	python scripts/make_sample.py
figures:
	python scripts/make_figures.py
evals:        ## score the assistant
	python -m assistant.evals.run_evals --athletes 40
notebooks:
	python scripts/make_notebooks.py
test:
	pytest -q
lint:
	ruff check .
app:
	streamlit run dashboard/app.py
all: data build train sample figures evals notebooks test
