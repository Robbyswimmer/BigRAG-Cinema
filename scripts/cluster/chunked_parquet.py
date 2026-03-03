#!/usr/bin/env python3
"""Memory-efficient parquet preparation that processes data in chunks.

Reads the JSONL in chunks, loads the corresponding embedding slice via
memory-mapped .npy, and writes each chunk as a separate parquet part file.
Spark reads a directory of parquet files natively as one DataFrame.

Usage:
    python scripts/cluster/chunked_parquet.py \
        --data-path data/raw/raw/review_categories/Video_Games.jsonl \
        --embeddings-path data/embeddings/Video_Games_embeddings.npy \
        --output-path data/parquet/Video_Games.parquet \
        --chunk-size 200000
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

from bigrag.data.schema import COL_EMBEDDING, COL_TEXT, JSONL_DROP_COLUMNS
from bigrag.data.validator import validate_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Chunked parquet preparation")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Rows per chunk (default: 200000)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    emb_path = Path(args.embeddings_path)
    output_path = Path(args.output_path)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    if not emb_path.exists():
        print(f"ERROR: Embeddings file not found: {emb_path}")
        sys.exit(1)

    # Output is a directory of parquet part files
    output_path.mkdir(parents=True, exist_ok=True)

    # Memory-map the embeddings (read-only, no RAM cost)
    print(f"Memory-mapping embeddings from {emb_path} ...")
    embeddings = np.load(str(emb_path), mmap_mode="r")
    total_emb_rows = embeddings.shape[0]
    print(f"  Embeddings shape: {embeddings.shape}")

    emb_offset = 0
    chunk_num = 0
    rows_written = 0
    t_start = time.perf_counter()

    reader = pd.read_json(data_path, lines=True, convert_dates=False,
                          chunksize=args.chunk_size)

    for chunk_df in reader:
        chunk_num += 1

        # Drop unneeded columns
        for col in JSONL_DROP_COLUMNS:
            if col in chunk_df.columns:
                chunk_df = chunk_df.drop(columns=[col])

        # Validate (same as embedding step — drops same rows)
        cleaned_df, report = validate_dataframe(chunk_df)
        del chunk_df
        gc.collect()

        n_valid = len(cleaned_df)
        if n_valid == 0:
            print(f"  Chunk {chunk_num}: 0 valid rows, skipping")
            continue

        # Check we have enough embeddings left
        if emb_offset + n_valid > total_emb_rows:
            print(f"  WARNING: Embedding offset ({emb_offset} + {n_valid}) exceeds "
                  f"total embeddings ({total_emb_rows}). Truncating chunk.")
            n_valid = total_emb_rows - emb_offset
            cleaned_df = cleaned_df.iloc[:n_valid]
            if n_valid == 0:
                break

        # Attach embeddings from mmap slice
        emb_slice = np.array(embeddings[emb_offset:emb_offset + n_valid])
        cleaned_df[COL_EMBEDDING] = emb_slice.astype(np.float32).tolist()
        emb_offset += n_valid

        # Write as a parquet part file
        part_path = output_path / f"part_{chunk_num:04d}.parquet"
        cleaned_df.to_parquet(part_path, index=False)

        rows_written += n_valid
        del cleaned_df, emb_slice
        gc.collect()

        elapsed = time.perf_counter() - t_start
        rate = rows_written / elapsed if elapsed > 0 else 0
        print(f"  Chunk {chunk_num}: wrote {n_valid:,} rows "
              f"({rows_written:,} total, {rate:,.0f} rows/sec)")

    elapsed = time.perf_counter() - t_start
    size_mb = sum(f.stat().st_size for f in output_path.glob("*.parquet")) / 1e6
    print(f"\nDone: {rows_written:,} rows -> {output_path}/ "
          f"({size_mb:.1f} MB total, {chunk_num} part files) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
