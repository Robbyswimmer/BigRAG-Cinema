#!/usr/bin/env python3
"""Memory-efficient embedding generation that processes JSONL in chunks.

Writes each chunk's embeddings to a memory-mapped file on disk, so RAM
usage stays constant regardless of dataset size.

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


def count_lines(path):
    """Count lines without loading file into memory."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


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
    total_lines = count_lines(input_path)
    print(f"  Total lines: {total_lines:,}")

    # Pre-allocate a memory-mapped .npy file on disk.
    # This avoids holding all embeddings in RAM.
    print(f"Pre-allocating output file ({total_lines:,} x {EMBEDDING_DIM}) ...")
    mmap_path = output_path
    fp = np.lib.format.open_memmap(
        str(mmap_path), mode="w+", dtype=np.float32,
        shape=(total_lines, EMBEDDING_DIM),
    )
    # We'll track the actual write offset since validation may drop rows
    write_offset = 0

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

        n_texts = len(texts)
        print(f"  Chunk {chunk_num}: encoding {n_texts:,} texts "
              f"({rows_done + n_texts:,}/{total_lines:,}) ...")

        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Write directly to memory-mapped file
        fp[write_offset:write_offset + n_texts] = emb
        write_offset += n_texts
        rows_done += n_texts

        # Free chunk data from RAM
        del texts, emb
        gc.collect()

        elapsed = time.perf_counter() - t_start
        rate = rows_done / elapsed if elapsed > 0 else 0
        print(f"    {rows_done:,} rows done in {elapsed:.0f}s ({rate:,.0f} texts/sec)")

    # Flush memory-mapped file
    del fp
    gc.collect()

    # If rows were dropped during validation, truncate the file to actual size
    if write_offset < total_lines:
        print(f"Trimming output from {total_lines:,} to {write_offset:,} rows "
              f"({total_lines - write_offset:,} dropped during validation) ...")
        full = np.load(str(output_path), mmap_mode="r")
        trimmed = np.array(full[:write_offset])
        del full
        np.save(str(output_path), trimmed)
        del trimmed
        gc.collect()

    size_mb = output_path.stat().st_size / 1e6
    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {write_offset:,} embeddings -> {output_path} "
          f"({size_mb:.1f} MB) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
