#!/usr/bin/env bash
#SBATCH --job-name=bigrag-embed-all
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/embed_all_%j.out
#SBATCH --error=logs/embed_all_%j.err
#
# Generate embeddings for ALL 34 Amazon Reviews categories in one GPU job.
# Usage:  sbatch scripts/cluster/embed_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate bigrag

mkdir -p data/embeddings logs

CATEGORIES=(
    All_Beauty
    Amazon_Fashion
    Appliances
    Arts_Crafts_and_Sewing
    Automotive
    Baby_Products
    Beauty_and_Personal_Care
    Books
    CDs_and_Vinyl
    Cell_Phones_and_Accessories
    Clothing_Shoes_and_Jewelry
    Digital_Music
    Electronics
    Gift_Cards
    Grocery_and_Gourmet_Food
    Handmade_Products
    Health_and_Household
    Health_and_Personal_Care
    Home_and_Kitchen
    Industrial_and_Scientific
    Kindle_Store
    Magazine_Subscriptions
    Movies_and_TV
    Musical_Instruments
    Office_Products
    Patio_Lawn_and_Garden
    Pet_Supplies
    Software
    Sports_and_Outdoors
    Subscription_Boxes
    Tools_and_Home_Improvement
    Toys_and_Games
    Unknown
    Video_Games
)

TOTAL=${#CATEGORIES[@]}
echo "=== Embedding Generation: All $TOTAL categories ==="
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "GPUs   : ${CUDA_VISIBLE_DEVICES:-none}"
echo ""

for i in "${!CATEGORIES[@]}"; do
    cat="${CATEGORIES[$i]}"
    num=$((i + 1))
    INPUT="data/raw/raw/review_categories/${cat}.jsonl"
    OUTPUT="data/embeddings/${cat}_embeddings.npy"

    if [[ -f "$OUTPUT" ]]; then
        echo "[$num/$TOTAL] $cat — already exists, skipping"
        continue
    fi

    if [[ ! -f "$INPUT" ]]; then
        echo "[$num/$TOTAL] $cat — input not found ($INPUT), skipping"
        continue
    fi

    echo "[$num/$TOTAL] $cat — generating embeddings..."
    python scripts/generate_embeddings.py \
        --input-path "$INPUT" \
        --output-path "$OUTPUT"
    echo "  Done: $OUTPUT"
done

echo ""
echo "=== All embeddings complete ==="
ls -lh data/embeddings/
