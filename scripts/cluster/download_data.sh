#!/usr/bin/env bash
# Download Amazon Reviews 2023 JSONL files from Hugging Face.
# Usage: bash scripts/cluster/download_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p data/raw/raw/review_categories

eval "$(conda shell.bash hook)"
conda activate bigrag

pip install -q huggingface_hub

python -c "
from huggingface_hub import hf_hub_download

for cat in ['Digital_Music', 'All_Beauty', 'Video_Games']:
    print(f'Downloading {cat}...')
    hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename=f'raw/review_categories/{cat}.jsonl',
        repo_type='dataset',
        local_dir='data/raw',
    )
    print(f'  Done: {cat}')
print('All downloads complete.')
"

echo ""
echo "Verifying..."
ls -lh data/raw/raw/review_categories/
wc -l data/raw/raw/review_categories/*.jsonl
