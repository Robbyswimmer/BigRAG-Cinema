#!/usr/bin/env bash
#SBATCH --job-name=bigrag-parquet
#SBATCH --partition=batch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/parquet_%x_%j.out
#SBATCH --error=logs/parquet_%x_%j.err
#
# Join embeddings + raw data into a Parquet file.
# Usage:  sbatch scripts/cluster/02_prepare_parquet.sh <DATASET>
#   e.g.  sbatch scripts/cluster/02_prepare_parquet.sh Digital_Music

set -euo pipefail

DATASET="${1:?Usage: sbatch $0 <DATASET>  (e.g. Digital_Music, All_Beauty, Video_Games)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# Paths
DATA_JSONL="data/raw/raw/review_categories/${DATASET}.jsonl"
EMBEDDINGS_NPY="data/embeddings/${DATASET}_embeddings.npy"
OUTPUT_PQ="data/parquet/${DATASET}.parquet"

echo "=== Parquet Preparation: $DATASET ==="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : $(hostname)"
echo "Data       : $DATA_JSONL"
echo "Embeddings : $EMBEDDINGS_NPY"
echo "Output     : $OUTPUT_PQ"
echo ""

# Skip if already done
if [[ -f "$OUTPUT_PQ" ]] || [[ -d "$OUTPUT_PQ" ]]; then
    echo "Parquet already exists at $OUTPUT_PQ — skipping."
    exit 0
fi

# Validate inputs
for f in "$DATA_JSONL" "$EMBEDDINGS_NPY"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f" >&2
        exit 1
    fi
done

# Activate environment
eval "$(conda shell.bash hook)"
conda activate bigrag

mkdir -p "$(dirname "$OUTPUT_PQ")"

python scripts/prepare_parquet.py \
    --data-path "$DATA_JSONL" \
    --embeddings-path "$EMBEDDINGS_NPY" \
    --output-path "$OUTPUT_PQ"

echo ""
echo "Done. Parquet saved to $OUTPUT_PQ"
