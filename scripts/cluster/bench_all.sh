#!/usr/bin/env bash
#SBATCH --job-name=bigrag-bench-all
#SBATCH --partition=batch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --chdir=/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema
#SBATCH --output=/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema/logs/bench_all_%j.out
#SBATCH --error=/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema/logs/bench_all_%j.err
#
# Run benchmarks for ALL categories that have parquet data and bench configs.
# Auto-generates configs for any new datasets before running.
# Skips categories that already have results.
#
# Usage:  sbatch scripts/cluster/bench_all.sh

set -euo pipefail

PROJECT_DIR="/data/SalmanAsif/RobbyMoseley/rag/BigRAG-Cinema"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate bigrag

mkdir -p results/raw_metrics logs

# Use the SLURM-node Spark profile
export BIGRAG_CLUSTER_PROFILE="conf/cluster_profiles/slurm_node.yaml"

echo "=== BigRAG Benchmark: All Categories ==="
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "CPUs   : ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory : ${SLURM_MEM_PER_NODE:-unknown}MB"
echo ""

# Step 1: Auto-generate configs for any datasets that don't have one yet
echo "--- Generating benchmark configs ---"
python scripts/cluster/generate_bench_configs.py
echo ""

# Step 2: Run benchmarks for each config
DONE=0
SKIPPED=0
FAILED=0
TOTAL=0

for CONFIG in conf/bench_*.yaml; do
    # Extract dataset name from config filename
    BASENAME="$(basename "$CONFIG" .yaml)"
    DATASET="${BASENAME#bench_}"
    TOTAL=$((TOTAL + 1))

    RESULTS_DIR="results/raw_metrics/${DATASET}"

    # Skip if results already exist
    if [[ -f "${RESULTS_DIR}/benchmark_results.json" ]]; then
        echo "[$TOTAL] $DATASET — results exist, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Check if parquet data exists
    PARQUET_PATH="data/parquet/$(python -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c['dataset']['source_path'].split('/')[-1])
" 2>/dev/null || echo "${DATASET}.parquet")"

    # Try both capitalized and lowercase parquet paths
    FOUND_PARQUET=""
    for PQ_CANDIDATE in "data/parquet/"*; do
        PQ_BASE="$(basename "$PQ_CANDIDATE")"
        PQ_LOWER="$(echo "$PQ_BASE" | tr '[:upper:]' '[:lower:]')"
        if [[ "$PQ_LOWER" == "${DATASET}.parquet" ]]; then
            FOUND_PARQUET="$PQ_CANDIDATE"
            break
        fi
    done

    if [[ -z "$FOUND_PARQUET" ]]; then
        echo "[$TOTAL] $DATASET — no parquet data found, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$TOTAL] $DATASET — running benchmarks..."
    echo "  Config:  $CONFIG"
    echo "  Parquet: $FOUND_PARQUET"

    if python scripts/run_benchmarks.py --config "$CONFIG"; then
        echo "  Done: results in $RESULTS_DIR/"
        DONE=$((DONE + 1))
    else
        echo "  FAILED: $DATASET (continuing with next)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== Benchmark run complete ==="
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Total:   $TOTAL"
echo ""
echo "Results in:"
ls -d results/raw_metrics/*/ 2>/dev/null || echo "  (none)"
