# BigRAG Cinema

PySpark-based system for benchmarking hybrid query execution strategies for Retrieval-Augmented Generation (RAG) workloads on the Amazon Reviews 2023 dataset.

## Overview

This project compares four query execution strategies for combining metadata filtering with vector similarity search:

| Strategy | Description |
|----------|-------------|
| **Filter-First** | Apply metadata filters, then vector search on the reduced subset |
| **Vector-First** | Run vector similarity search, then filter the top candidates |
| **Hybrid Parallel** | Execute filter and vector search concurrently, merge results |
| **Adaptive** | Heuristic-based strategy selection at runtime based on query characteristics |

## Project Structure

```
BigRAG Cinema/
├── conf/               # Spark, experiment, and logging configuration
├── data/               # Raw, processed, embeddings, and Parquet data (git-ignored)
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/bigrag/         # Core Python package
│   ├── data/           # Data download, validation, embedding, Parquet writing
│   ├── engine/         # SparkSession, vector search, metadata filters, query API
│   ├── strategies/     # Four execution strategies + registry
│   ├── benchmark/      # Workload generation, metrics, experiment orchestration
│   ├── analysis/       # Statistics, plotting, report table generation
│   └── utils/          # Config loading, logging setup, I/O helpers
├── scripts/            # CLI wrappers for pipeline stages
├── tests/              # Unit and integration tests
├── results/            # Benchmark outputs: metrics, figures, tables (git-ignored)
└── docs/               # Proposal, report, presentation
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd "BigRAG Cinema"

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Usage

```bash
# Run the full pipeline
make all

# Or run individual stages
make data          # Download the dataset
make embeddings    # Generate sentence-transformer embeddings
make parquet       # Write Parquet files at various scale fractions
make bench         # Run benchmark experiments
make plots         # Generate result visualizations
make test          # Run tests
```

## Dataset

Amazon Reviews 2023 dataset from Kaggle. Embeddings are generated using the `all-MiniLM-L6-v2` sentence-transformer model (384-dimensional vectors).

## Requirements

- Python >= 3.10
- Apache Spark >= 3.5
- See `requirements.txt` for full dependency list

## Team

- Robby Moseley
