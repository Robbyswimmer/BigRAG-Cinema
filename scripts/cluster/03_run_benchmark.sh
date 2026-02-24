#!/usr/bin/env bash
#SBATCH --job-name=bigrag-bench
#SBATCH --partition=batch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=logs/bench_%x_%j.out
#SBATCH --error=logs/bench_%x_%j.err
#
# Run benchmarks for a dataset using Spark local[*] mode.
# Usage:  sbatch scripts/cluster/03_run_benchmark.sh <DATASET>
#   e.g.  sbatch scripts/cluster/03_run_benchmark.sh Digital_Music
#
# Optional env vars (set before sbatch or via --export):
#   NUM_QUERIES  — override number of queries (default: from config)
#   NUM_REPS     — override number of repetitions (default: from config)

set -euo pipefail

DATASET="${1:?Usage: sbatch $0 <DATASET>  (e.g. Digital_Music, All_Beauty, Video_Games)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# Map dataset name to config file (lowercase with underscores)
DATASET_LOWER="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
CONFIG="conf/bench_${DATASET_LOWER}.yaml"

echo "=== Benchmark: $DATASET ==="
echo "Job ID  : ${SLURM_JOB_ID:-local}"
echo "Node    : $(hostname)"
echo "CPUs    : ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory  : ${SLURM_MEM_PER_NODE:-unknown}MB"
echo "Config  : $CONFIG"
echo ""

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    exit 1
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate bigrag

# Configure Spark to use the SLURM-node cluster profile
export BIGRAG_CLUSTER_PROFILE="conf/cluster_profiles/slurm_node.yaml"

# Build CLI args
BENCH_ARGS=(--config "$CONFIG")

if [[ -n "${NUM_QUERIES:-}" ]]; then
    echo "Overriding num_queries=$NUM_QUERIES"
    BENCH_ARGS+=(--num-queries "$NUM_QUERIES")
fi

if [[ -n "${NUM_REPS:-}" ]]; then
    echo "Overriding num_reps=$NUM_REPS"
    BENCH_ARGS+=(--num-reps "$NUM_REPS")
fi

echo "Running: python scripts/run_benchmarks.py ${BENCH_ARGS[*]}"
echo ""

python scripts/run_benchmarks.py "${BENCH_ARGS[@]}"

echo ""
echo "Done. Results in results/raw_metrics/${DATASET_LOWER}/"
