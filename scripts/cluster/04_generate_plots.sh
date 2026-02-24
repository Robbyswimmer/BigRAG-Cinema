#!/usr/bin/env bash
#SBATCH --job-name=bigrag-plots
#SBATCH --partition=batch
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --output=logs/plots_%x_%j.out
#SBATCH --error=logs/plots_%x_%j.err
#
# Generate plots for a single dataset or cross-dataset comparison.
# Usage:
#   sbatch scripts/cluster/04_generate_plots.sh Digital_Music
#   sbatch scripts/cluster/04_generate_plots.sh comparison

set -euo pipefail

DATASET="${1:?Usage: sbatch $0 <DATASET|comparison>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate bigrag

echo "=== Plot Generation: $DATASET ==="
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo ""

if [[ "$DATASET" == "comparison" ]]; then
    # Cross-dataset comparison plots
    echo "Generating cross-dataset comparison plots ..."
    python -c "
from bigrag.analysis.plotting import generate_cross_dataset_plots
generate_cross_dataset_plots('results/raw_metrics', 'results/figures/comparison')
"
    echo "Done. Figures in results/figures/comparison/"
else
    DATASET_LOWER="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
    METRICS_DIR="results/raw_metrics/${DATASET_LOWER}"
    OUTPUT_DIR="results/figures/${DATASET_LOWER}"

    if [[ ! -d "$METRICS_DIR" ]]; then
        echo "ERROR: Metrics directory not found: $METRICS_DIR" >&2
        exit 1
    fi

    echo "Generating plots for $DATASET ..."
    mkdir -p "$OUTPUT_DIR"

    python scripts/generate_plots.py \
        --metrics-dir "$METRICS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --label "$DATASET"

    echo "Done. Figures in $OUTPUT_DIR/"
fi
