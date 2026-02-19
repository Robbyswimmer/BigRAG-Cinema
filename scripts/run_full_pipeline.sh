#!/usr/bin/env bash
# run_full_pipeline.sh
# Runs the full BigRAG Cinema pipeline for a single JSONL category file.
#
# Usage:
#   bash scripts/run_full_pipeline.sh data/raw/raw/review_categories/All_Beauty.jsonl
#   bash scripts/run_full_pipeline.sh data/raw/raw/review_categories/Video_Games.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-jsonl>"
    echo "Example: $0 data/raw/raw/review_categories/All_Beauty.jsonl"
    exit 1
fi

INPUT_FILE="$1"
BASENAME="$(basename "$INPUT_FILE" .jsonl)"

EMBEDDINGS_PATH="$PROJECT_DIR/data/embeddings/${BASENAME}_embeddings.npy"
PARQUET_PATH="$PROJECT_DIR/data/parquet/${BASENAME}.parquet"

echo "=== BigRAG Cinema — Full Pipeline ==="
echo "Project directory: $PROJECT_DIR"
echo "Input file:        $INPUT_FILE"
echo "Category:          $BASENAME"
echo ""

echo "[1/4] Generating embeddings..."
python "$SCRIPT_DIR/generate_embeddings.py" \
    --input-path "$INPUT_FILE" \
    --output-path "$EMBEDDINGS_PATH"

echo "[2/4] Preparing Parquet files..."
python "$SCRIPT_DIR/prepare_parquet.py" \
    --data-path "$INPUT_FILE" \
    --embeddings-path "$EMBEDDINGS_PATH" \
    --output-path "$PARQUET_PATH"

echo "[3/4] Running benchmarks..."
python "$SCRIPT_DIR/run_benchmarks.py"

echo "[4/4] Generating plots..."
python "$SCRIPT_DIR/generate_plots.py"

echo ""
echo "=== Pipeline complete for $BASENAME ==="
