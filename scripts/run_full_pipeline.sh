#!/usr/bin/env bash
# run_full_pipeline.sh
# Runs the full BigRAG Cinema pipeline end-to-end:
#   1. Download dataset
#   2. Generate embeddings
#   3. Prepare Parquet files
#   4. Run benchmarks
#   5. Generate plots

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== BigRAG Cinema — Full Pipeline ==="
echo "Project directory: $PROJECT_DIR"
echo ""

echo "[1/5] Downloading dataset..."
python "$SCRIPT_DIR/download_dataset.py" "$@"

echo "[2/5] Generating embeddings..."
python "$SCRIPT_DIR/generate_embeddings.py"

echo "[3/5] Preparing Parquet files..."
python "$SCRIPT_DIR/prepare_parquet.py"

echo "[4/5] Running benchmarks..."
python "$SCRIPT_DIR/run_benchmarks.py"

echo "[5/5] Generating plots..."
python "$SCRIPT_DIR/generate_plots.py"

echo ""
echo "=== Pipeline complete ==="
