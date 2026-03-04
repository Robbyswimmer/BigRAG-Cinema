#!/usr/bin/env bash
#SBATCH --job-name=bigrag-parquet-all
#SBATCH --partition=batch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --chdir=/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema
#SBATCH --output=/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema/logs/parquet_all_%j.out
#SBATCH --error=/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema/logs/parquet_all_%j.err
#
# Prepare parquet files for ALL categories that have embeddings.
# Skips categories missing JSONL or embeddings, and skips already-done parquets.
# Usage:  sbatch scripts/cluster/parquet_all.sh

set -euo pipefail

PROJECT_DIR="/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate bigrag

mkdir -p data/parquet logs

echo "=== Parquet Preparation: All Available Categories ==="
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo ""

# Auto-discover categories from available embeddings
DONE=0
SKIPPED=0
FAILED=0
TOTAL=0

for EMB_FILE in data/embeddings/*_embeddings.npy; do
    # Extract category name from filename
    BASENAME="$(basename "$EMB_FILE")"
    CATEGORY="${BASENAME%_embeddings.npy}"
    TOTAL=$((TOTAL + 1))

    JSONL="data/raw/raw/review_categories/${CATEGORY}.jsonl"
    OUTPUT="data/parquet/${CATEGORY}.parquet"

    # Skip if parquet already exists
    if [[ -f "$OUTPUT" ]] || [[ -d "$OUTPUT" ]]; then
        echo "[$TOTAL] $CATEGORY — parquet exists, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if JSONL is missing
    if [[ ! -f "$JSONL" ]]; then
        echo "[$TOTAL] $CATEGORY — JSONL not found ($JSONL), skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$TOTAL] $CATEGORY — preparing parquet..."
    if python scripts/cluster/chunked_parquet.py \
        --data-path "$JSONL" \
        --embeddings-path "$EMB_FILE" \
        --output-path "$OUTPUT" \
        --chunk-size 200000; then
        echo "  Done: $OUTPUT"
        DONE=$((DONE + 1))
    else
        echo "  FAILED: $CATEGORY (continuing with next)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== Parquet preparation complete ==="
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Total:   $TOTAL"
echo ""
ls -lhd data/parquet/*/  2>/dev/null || ls -lh data/parquet/ 2>/dev/null || echo "No parquet files found"
