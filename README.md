# BigRAG Cinema: Optimizing Hybrid Query Execution for RAG Systems at Scale

**Course:** CS 226 — Big Data Management (Winter 2026)
**Group Name:** Team Big RAG + CineMetrics
**Group Number:** XX *(replace with your group number)*

## Team Members

| Name | Email | Student ID |
|------|-------|------------|
| Robert Moseley | rmose009@ucr.edu | 862604323 |
| Muntashir Al-Islam | malis002@ucr.edu | 862546294 |
| Amirhossein Saririghavee | asari009@ucr.edu | 862377262 |
| Dylan Tang | dtang041@ucr.edu | 862351306 |
| Bufan Zhou | bzhou059@ucr.edu | 862547556 |

## Overview

A PySpark-based benchmarking system that evaluates four hybrid query execution strategies for Retrieval-Augmented Generation (RAG) workloads on the Amazon Reviews 2023 dataset. The system combines structured SQL-style metadata filters (time ranges, ratings, user IDs) with vector similarity search (384-dimensional sentence embeddings) and measures latency, throughput, recall, and scalability across five product categories totaling 20 million records.

### Execution Strategies

| Strategy | Description |
|----------|-------------|
| **Filter-First** | Apply metadata filters first, then vector search on the reduced subset |
| **Vector-First** | Run vector similarity search on the full dataset, then filter the top candidates |
| **Hybrid Parallel** | Execute filter and vector branches concurrently, merge and deduplicate results |
| **Adaptive** | Estimate filter selectivity at runtime and delegate to the best strategy |

## Project Structure

```
BigRAG Cinema/
├── conf/                   # Configuration files
│   ├── cluster_profiles/   # Spark cluster YAML profiles (local, SLURM)
│   ├── experiment_config.yaml
│   └── bench_*.yaml        # Per-dataset benchmark configs
├── src/bigrag/             # Core Python package
│   ├── data/               # Dataset download, validation, embedding, Parquet I/O
│   ├── engine/             # SparkSession factory, vector search, metadata filters
│   ├── strategies/         # Four execution strategies + registry
│   ├── benchmark/          # Workload generation, metrics collection, experiment runner
│   ├── analysis/           # Statistics, plotting, LaTeX table generation
│   └── utils/              # Config loading, logging, I/O helpers
├── scripts/                # CLI entry points for each pipeline stage
│   ├── cluster/            # SLURM cluster batch scripts
│   ├── compute_recall.py   # Recall@K computation
│   ├── generate_tables.py  # LaTeX table generation
│   └── generate_plots.py   # Figure generation
├── tests/                  # Unit and integration tests
├── results/                # Benchmark outputs (metrics JSON, figures, tables)
├── Makefile                # Build automation
├── pyproject.toml          # Python package configuration
└── requirements.txt        # Dependency list
```

## Prerequisites

- Python >= 3.9
- Java 11 or 17 (required by Apache Spark)
- Apache Spark >= 3.5

## Setup and Installation

```bash
# 1. Navigate to the project directory
cd "BigRAG Cinema"

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install the package and all dependencies
pip install -e ".[dev]"

# Alternatively, install from requirements.txt
pip install -r requirements.txt
```

## How to Run

### Full Pipeline (Makefile)

```bash
make all       # Runs: install → data → embeddings → parquet → bench → plots
```

### Individual Stages

```bash
# Step 1: Download the Amazon Reviews 2023 dataset
python scripts/download_dataset.py

# Step 2: Generate sentence-transformer embeddings (384-dim, all-MiniLM-L6-v2)
python scripts/generate_embeddings.py

# Step 3: Write Parquet files with embeddings at various data fractions
python scripts/prepare_parquet.py

# Step 4: Run benchmark experiments (all 4 strategies x 5 fractions)
python scripts/run_benchmarks.py

# Step 5: Compute Recall@K (uses filter-first as ground truth)
python scripts/compute_recall.py --data-dir data/parquet --num-queries 15

# Step 6: Generate LaTeX tables from benchmark results
python scripts/generate_tables.py --metrics-dir results/raw_metrics --output-dir results/tables

# Step 7: Generate figures and plots
python scripts/generate_plots.py --metrics-dir results/raw_metrics --output-dir results/figures
```

### Running on a SLURM Cluster

The `scripts/cluster/` directory contains batch scripts for SLURM-managed clusters:

```bash
# Set up the environment on the cluster node
bash scripts/cluster/setup_env.sh

# Generate per-dataset benchmark configs
python scripts/cluster/generate_bench_configs.py

# Submit all benchmark jobs
bash scripts/cluster/submit_all.sh
```

### Running Tests

```bash
pytest tests/ -v
```

## Configuration

Spark settings are defined in YAML cluster profiles under `conf/cluster_profiles/`. The default profile (`local.yaml`) configures Spark in `local[*]` mode. Experiment parameters (strategies, fractions, query counts) are set in `conf/experiment_config.yaml` or per-dataset configs (`conf/bench_*.yaml`).

Key environment variable:
- `BIGRAG_CLUSTER_PROFILE` — path to a Spark cluster profile YAML (overrides config default)

## Datasets

Amazon Reviews 2023 (McAuley et al., 2013). Five product categories used in benchmarking:

| Category | Rows |
|----------|------|
| All Beauty | 693,548 |
| Amazon Fashion | 2,473,302 |
| Appliances | 2,103,927 |
| Arts & Crafts | 8,859,715 |
| Baby Products | 5,962,394 |
| **Total** | **20,092,886** |

Embeddings are generated using the `all-MiniLM-L6-v2` sentence-transformer model (384-dimensional vectors, normalized).

## Author Contributions

**Robert Moseley:** Project lead and system architect. Designed and implemented the core benchmarking framework (`src/bigrag/benchmark/`), including the experiment runner, metrics collection with Spark StatusStore integration, and the recall computation pipeline. Built the analysis and reporting system (`src/bigrag/analysis/`) for generating LaTeX tables and figures. Wrote and maintained the Makefile, CI scripts, and cluster deployment configurations. Authored the project report sections on system architecture, experimental evaluation, and results analysis.

**Muntashir Al-Islam:** Implemented the data ingestion pipeline (`src/bigrag/data/`), including the dataset downloader, data validator, and Parquet writer. Developed the sentence-transformer embedding pipeline with GPU/MPS acceleration support. Contributed to the project report sections on dataset preparation and embedding generation.

**Amirhossein Saririghavee:** Implemented the four query execution strategies (`src/bigrag/strategies/`): Filter-First, Vector-First, Hybrid Parallel, and Adaptive. Designed the strategy registry and the adaptive selectivity estimation logic. Contributed to the project report sections on methodology and strategy design.

**Dylan Tang:** Built the Spark engine layer (`src/bigrag/engine/`), including the SparkSession factory, vector similarity search (brute-force cosine), and metadata filter expression builder. Handled Spark configuration tuning and cluster profile management. Contributed to the project report sections on Spark configuration and query optimization.

**Bufan Zhou:** Developed the workload generator (`src/bigrag/benchmark/workload_generator.py`) with configurable selectivity buckets. Created the plotting and visualization module (`src/bigrag/analysis/plotting.py`). Managed SLURM cluster job submission scripts. Contributed to the project report sections on workload design and related work.
