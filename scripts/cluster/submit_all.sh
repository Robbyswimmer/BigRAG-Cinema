#!/usr/bin/env bash
# Master submission script — submits the full BigRAG Cinema pipeline as
# SLURM jobs with dependency chains.
#
# Usage:
#   bash scripts/cluster/submit_all.sh                 # all 3 datasets
#   bash scripts/cluster/submit_all.sh --skip-video-games
#   bash scripts/cluster/submit_all.sh --skip-embeddings --skip-parquet
#
# Flags:
#   --skip-video-games   Exclude Video_Games (largest dataset)
#   --skip-embeddings    Skip embedding generation (data already exists)
#   --skip-parquet       Skip parquet preparation (data already exists)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Parse flags ───────────────────────────────────────────────────────
SKIP_VG=false
SKIP_EMBEDDINGS=false
SKIP_PARQUET=false

for arg in "$@"; do
    case "$arg" in
        --skip-video-games) SKIP_VG=true ;;
        --skip-embeddings)  SKIP_EMBEDDINGS=true ;;
        --skip-parquet)     SKIP_PARQUET=true ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── Datasets ──────────────────────────────────────────────────────────
DATASETS=("Digital_Music" "All_Beauty")
if [[ "$SKIP_VG" == false ]]; then
    DATASETS+=("Video_Games")
fi

echo "=== BigRAG Cinema — SLURM Pipeline Submission ==="
echo "Datasets       : ${DATASETS[*]}"
echo "Skip embeddings: $SKIP_EMBEDDINGS"
echo "Skip parquet   : $SKIP_PARQUET"
echo ""

# Ensure logs directory exists
mkdir -p logs

# Helper to extract job ID from sbatch output ("Submitted batch job 12345")
parse_jobid() {
    local output="$1"
    echo "$output" | awk '{print $NF}'
}

# ── Stage 1: Embedding generation (GPU) ──────────────────────────────
declare -A EMB_JOBS

if [[ "$SKIP_EMBEDDINGS" == false ]]; then
    echo "--- Stage 1: Embedding generation (GPU) ---"
    for ds in "${DATASETS[@]}"; do
        out=$(sbatch --job-name="embed-${ds}" \
                     scripts/cluster/01_generate_embeddings.sh "$ds")
        jid=$(parse_jobid "$out")
        EMB_JOBS[$ds]="$jid"
        echo "  $ds : job $jid"
    done
    echo ""
fi

# ── Stage 2: Parquet preparation (CPU) ───────────────────────────────
declare -A PQ_JOBS

if [[ "$SKIP_PARQUET" == false ]]; then
    echo "--- Stage 2: Parquet preparation (CPU) ---"
    for ds in "${DATASETS[@]}"; do
        DEP_FLAG=""
        if [[ -n "${EMB_JOBS[$ds]:-}" ]]; then
            DEP_FLAG="--dependency=afterok:${EMB_JOBS[$ds]}"
        fi
        out=$(sbatch --job-name="parquet-${ds}" \
                     $DEP_FLAG \
                     scripts/cluster/02_prepare_parquet.sh "$ds")
        jid=$(parse_jobid "$out")
        PQ_JOBS[$ds]="$jid"
        echo "  $ds : job $jid  ${DEP_FLAG:+(depends on ${EMB_JOBS[$ds]:-})}"
    done
    echo ""
fi

# ── Stage 3: Benchmarks (CPU, big node) ──────────────────────────────
declare -A BENCH_JOBS

echo "--- Stage 3: Benchmarks (CPU) ---"
for ds in "${DATASETS[@]}"; do
    DEP_FLAG=""
    if [[ -n "${PQ_JOBS[$ds]:-}" ]]; then
        DEP_FLAG="--dependency=afterok:${PQ_JOBS[$ds]}"
    fi
    out=$(sbatch --job-name="bench-${ds}" \
                 $DEP_FLAG \
                 scripts/cluster/03_run_benchmark.sh "$ds")
    jid=$(parse_jobid "$out")
    BENCH_JOBS[$ds]="$jid"
    echo "  $ds : job $jid  ${DEP_FLAG:+(depends on ${PQ_JOBS[$ds]:-})}"
done
echo ""

# ── Stage 4: Plot generation (lightweight CPU) ───────────────────────
echo "--- Stage 4: Plot generation ---"
ALL_BENCH_IDS=""

for ds in "${DATASETS[@]}"; do
    DEP_FLAG="--dependency=afterok:${BENCH_JOBS[$ds]}"
    out=$(sbatch --job-name="plots-${ds}" \
                 $DEP_FLAG \
                 scripts/cluster/04_generate_plots.sh "$ds")
    jid=$(parse_jobid "$out")
    echo "  $ds plots : job $jid  (depends on ${BENCH_JOBS[$ds]})"
    ALL_BENCH_IDS="${ALL_BENCH_IDS:+${ALL_BENCH_IDS}:}${BENCH_JOBS[$ds]}"
done

# Cross-dataset comparison (depends on ALL benchmarks)
COMP_DEP="--dependency=afterok:${ALL_BENCH_IDS}"
out=$(sbatch --job-name="plots-comparison" \
             $COMP_DEP \
             scripts/cluster/04_generate_plots.sh comparison)
jid=$(parse_jobid "$out")
echo "  comparison : job $jid  (depends on all benchmarks)"
echo ""

# ── Summary ───────────────────────────────────────────────────────────
echo "=== Pipeline submitted ==="
echo ""
echo "Dependency chain per dataset:"
for ds in "${DATASETS[@]}"; do
    chain=""
    [[ -n "${EMB_JOBS[$ds]:-}" ]] && chain+="embed(${EMB_JOBS[$ds]}) -> "
    [[ -n "${PQ_JOBS[$ds]:-}" ]] && chain+="parquet(${PQ_JOBS[$ds]}) -> "
    chain+="bench(${BENCH_JOBS[$ds]}) -> plots"
    echo "  $ds : $chain"
done
echo ""
echo "Monitor with:  squeue -u $USER"
echo "Logs in:       logs/"
