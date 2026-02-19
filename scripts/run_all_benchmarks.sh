#!/usr/bin/env bash
# run_all_benchmarks.sh
# Runs the full benchmark suite across all datasets and generates plots.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh          # run everything
#   bash scripts/run_all_benchmarks.sh --skip-video-games   # skip the large dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SKIP_VG=false
for arg in "$@"; do
    if [ "$arg" = "--skip-video-games" ]; then
        SKIP_VG=true
    fi
done

echo "============================================================"
echo "  BigRAG Cinema — Full Benchmark Suite"
echo "============================================================"
echo ""

# ── Step 0: Video_Games data prep (if needed) ──────────────────────────
VG_JSONL="data/raw/raw/review_categories/Video_Games.jsonl"
VG_EMB="data/embeddings/Video_Games_embeddings.npy"
VG_PQ="data/parquet/Video_Games.parquet"

if [ "$SKIP_VG" = false ]; then
    if [ ! -f "$VG_EMB" ]; then
        echo "[0a] Generating Video_Games embeddings (this will take a while)..."
        python scripts/generate_embeddings.py \
            --input-path "$VG_JSONL" \
            --output-path "$VG_EMB"
        echo ""
    else
        echo "[0a] Video_Games embeddings already exist, skipping."
    fi

    if [ ! -f "$VG_PQ" ]; then
        echo "[0b] Preparing Video_Games parquet..."
        python scripts/prepare_parquet.py \
            --data-path "$VG_JSONL" \
            --embeddings-path "$VG_EMB" \
            --output-path "$VG_PQ"
        echo ""
    else
        echo "[0b] Video_Games parquet already exists, skipping."
    fi
fi

# ── Step 1: Digital_Music benchmark (~15 min) ──────────────────────────
echo ""
echo "============================================================"
echo "  [1/3] Benchmarking Digital_Music (130K rows)"
echo "        50 queries x 3 reps = 150 per strategy/fraction"
echo "============================================================"
python scripts/run_benchmarks.py --config conf/bench_digital_music.yaml
echo ""

# ── Step 2: All_Beauty benchmark (~2 hours) ────────────────────────────
echo "============================================================"
echo "  [2/3] Benchmarking All_Beauty (693K rows)"
echo "        50 queries x 3 reps = 150 per strategy/fraction"
echo "============================================================"
python scripts/run_benchmarks.py --config conf/bench_all_beauty.yaml
echo ""

# ── Step 3: Video_Games benchmark (~3-4 hours, optional) ───────────────
if [ "$SKIP_VG" = false ]; then
    echo "============================================================"
    echo "  [3/3] Benchmarking Video_Games (4.6M rows)"
    echo "        50 queries x 1 rep = 50 per strategy/fraction"
    echo "============================================================"
    python scripts/run_benchmarks.py --config conf/bench_video_games.yaml
    echo ""
else
    echo "[3/3] Skipping Video_Games (--skip-video-games flag set)"
    echo ""
fi

# ── Step 4: Generate per-dataset figures ───────────────────────────────
echo "============================================================"
echo "  Generating figures..."
echo "============================================================"

python scripts/generate_plots.py \
    --metrics-dir results/raw_metrics/digital_music \
    --output-dir results/figures/digital_music \
    --label Digital_Music

python scripts/generate_plots.py \
    --metrics-dir results/raw_metrics/all_beauty \
    --output-dir results/figures/all_beauty \
    --label All_Beauty

if [ "$SKIP_VG" = false ] && [ -f "results/raw_metrics/video_games/benchmark_results.json" ]; then
    python scripts/generate_plots.py \
        --metrics-dir results/raw_metrics/video_games \
        --output-dir results/figures/video_games \
        --label Video_Games
fi

# ── Step 5: Generate cross-dataset comparison ──────────────────────────
echo ""
echo "============================================================"
echo "  Generating cross-dataset comparison..."
echo "============================================================"
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from bigrag.analysis.plotting import generate_cross_dataset_plots
generate_cross_dataset_plots('results/raw_metrics', 'results/figures/comparison')
"

echo ""
echo "============================================================"
echo "  All done!"
echo ""
echo "  Results:   results/raw_metrics/{digital_music,all_beauty,video_games}/"
echo "  Figures:   results/figures/{digital_music,all_beauty,video_games,comparison}/"
echo "  Tables:    results/tables/"
echo "============================================================"
