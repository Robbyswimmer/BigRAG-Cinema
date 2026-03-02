#!/usr/bin/env python3
"""Memory-efficient embedding generation that processes JSONL in chunks.

Saves each chunk's embeddings to a separate temp file, then merges them
into the final .npy using memory-mapped I/O. Peak RAM usage is ~2-3 GB
regardless of dataset size.

Usage:
    python scripts/cluster/chunked_embed.py \
        --input-path data/raw/raw/review_categories/Video_Games.jsonl \
        --output-path data/embeddings/Video_Games_embeddings.npy \
        --chunk-size 200000
"""

import argparse
import gc
import shutil
import sys
import tempfile
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
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Rows per chunk (default: 200000)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size for sentence-transformers")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    return parser.parse_args()


def count_lines(path):
    """Count lines without loading file into memory."""
    count = 0
    with open(path, "rb") as f:
        buf_size = 1024 * 1024
        buf = f.raw.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = f.raw.read(buf_size)
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

    # Create temp directory for chunk files
    tmp_dir = Path(tempfile.mkdtemp(prefix="bigrag_embed_"))
    print(f"  Temp dir: {tmp_dir}")

    chunk_files = []
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

        # Free the dataframe immediately
        del chunk_df, cleaned_df, report
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
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Save chunk to temp file and free from RAM
        chunk_path = tmp_dir / f"chunk_{chunk_num:04d}.npy"
        np.save(str(chunk_path), emb)
        chunk_files.append((chunk_path, n_texts))
        rows_done += n_texts

        del texts, emb
        gc.collect()

        elapsed = time.perf_counter() - t_start
        rate = rows_done / elapsed if elapsed > 0 else 0
        print(f"    {rows_done:,} rows done in {elapsed:.0f}s ({rate:,.0f} texts/sec)")

    # ── Merge chunk files into final .npy via memory-mapped write ─────
    print(f"\nMerging {len(chunk_files)} chunk files ({rows_done:,} total rows) ...")

    # Create the final memory-mapped output
    fp = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=np.float32,
        shape=(rows_done, EMBEDDING_DIM),
    )

    offset = 0
    for i, (chunk_path, n) in enumerate(chunk_files):
        # Load chunk via mmap (read-only, doesn't use RAM)
        chunk_data = np.load(str(chunk_path), mmap_mode="r")
        fp[offset:offset + n] = chunk_data
        offset += n
        del chunk_data

        # Delete temp file as we go to free disk space
        chunk_path.unlink()

        if (i + 1) % 50 == 0 or (i + 1) == len(chunk_files):
            print(f"    Merged {i + 1}/{len(chunk_files)} chunks")

    # Flush and close mmap
    del fp
    gc.collect()

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    size_mb = output_path.stat().st_size / 1e6
    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {rows_done:,} embeddings -> {output_path} "
          f"({size_mb:.1f} MB) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
