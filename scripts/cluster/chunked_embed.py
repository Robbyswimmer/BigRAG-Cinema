#!/usr/bin/env python3
"""Memory-efficient embedding generation that processes JSONL in chunks.

Reads the input file in chunks to avoid loading the entire dataset into RAM,
generates embeddings per chunk, and concatenates at the end.

Usage:
    python scripts/cluster/chunked_embed.py \
        --input-path data/raw/raw/review_categories/Video_Games.jsonl \
        --output-path data/embeddings/Video_Games_embeddings.npy \
        --chunk-size 500000
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigrag.data.schema import COL_TEXT, EMBEDDING_DIM, JSONL_DROP_COLUMNS
from bigrag.data.validator import validate_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Chunked embedding generation")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000,
                        help="Rows per chunk (default: 500000)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size for sentence-transformers")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model once
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{args.model_name}' on device '{device}' ...")
    model = SentenceTransformer(args.model_name, device=device)

    # Count total lines for progress
    print(f"Counting lines in {input_path} ...")
    total_lines = sum(1 for _ in open(input_path))
    print(f"  Total lines: {total_lines:,}")

    # Process in chunks
    chunk_embeddings = []
    rows_done = 0
    chunk_num = 0
    t_start = time.perf_counter()

    reader = pd.read_json(input_path, lines=True, convert_dates=False,
                          chunksize=args.chunk_size)

    for chunk_df in reader:
        chunk_num += 1

        # Drop unneeded columns
        for col in JSONL_DROP_COLUMNS:
            if col in chunk_df.columns:
                chunk_df = chunk_df.drop(columns=[col])

        # Validate
        cleaned_df, report = validate_dataframe(chunk_df)
        texts = cleaned_df[COL_TEXT].astype(str).tolist()

        # Free the dataframe
        del chunk_df, cleaned_df
        gc.collect()

        if not texts:
            print(f"  Chunk {chunk_num}: 0 valid texts, skipping")
            continue

        print(f"  Chunk {chunk_num}: encoding {len(texts):,} texts "
              f"({rows_done + len(texts):,}/{total_lines:,}) ...")

        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        chunk_embeddings.append(emb)
        rows_done += len(texts)

        # Free texts
        del texts, emb
        gc.collect()

        elapsed = time.perf_counter() - t_start
        rate = rows_done / elapsed if elapsed > 0 else 0
        print(f"    {rows_done:,} rows done in {elapsed:.0f}s ({rate:,.0f} texts/sec)")

    # Concatenate all chunks
    print("Concatenating embeddings ...")
    all_embeddings = np.concatenate(chunk_embeddings, axis=0)
    del chunk_embeddings
    gc.collect()

    # Save
    np.save(output_path, all_embeddings)
    size_mb = output_path.stat().st_size / 1e6
    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {all_embeddings.shape[0]:,} embeddings -> {output_path} "
          f"({size_mb:.1f} MB) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
