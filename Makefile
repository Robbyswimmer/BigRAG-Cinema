.PHONY: install data embeddings parquet test bench plots clean all

install:
	pip install -e ".[dev]"

data:
	python scripts/download_dataset.py

embeddings:
	python scripts/generate_embeddings.py

parquet:
	python scripts/prepare_parquet.py

test:
	pytest tests/ -v

bench:
	python scripts/run_benchmarks.py

plots:
	python scripts/generate_plots.py

pipeline:
	bash scripts/run_full_pipeline.sh

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: install data embeddings parquet bench plots
