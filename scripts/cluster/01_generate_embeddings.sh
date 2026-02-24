#!/usr/bin/env bash
#SBATCH --job-name=bigrag-embed
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=logs/embed_%x_%j.out
#SBATCH --error=logs/embed_%x_%j.err
#
# Generate sentence-transformer embeddings for a dataset.
# Usage:  sbatch scripts/cluster/01_generate_embeddings.sh <DATASET>
#   e.g.  sbatch scripts/cluster/01_generate_embeddings.sh Digital_Music

set -euo pipefail

DATASET="${1:?Usage: sbatch $0 <DATASET>  (e.g. Digital_Music, All_Beauty, Video_Games)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# Paths
INPUT_JSONL="data/raw/raw/review_categories/${DATASET}.jsonl"
OUTPUT_NPY="data/embeddings/${DATASET}_embeddings.npy"

echo "=== Embedding Generation: $DATASET ==="
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Input    : $INPUT_JSONL"
echo "Output   : $OUTPUT_NPY"
echo ""

# Skip if already done
if [[ -f "$OUTPUT_NPY" ]]; then
    echo "Embeddings already exist at $OUTPUT_NPY — skipping."
    exit 0
fi

# Validate input
if [[ ! -f "$INPUT_JSONL" ]]; then
    echo "ERROR: Input file not found: $INPUT_JSONL" >&2
    exit 1
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate bigrag

mkdir -p "$(dirname "$OUTPUT_NPY")"

python scripts/generate_embeddings.py \
    --input-path "$INPUT_JSONL" \
    --output-path "$OUTPUT_NPY"

echo ""
echo "Done. Embeddings saved to $OUTPUT_NPY"
