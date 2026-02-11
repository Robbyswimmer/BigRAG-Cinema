#!/usr/bin/env python3
"""Generate sentence-transformer embeddings for the review dataset.

Thin CLI wrapper around bigrag.data.embedder.
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sentence-transformer embeddings in batches."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/raw/Reviews.csv",
        help="Path to the raw reviews CSV (default: data/raw/Reviews.csv)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/embeddings.npy",
        help="Path to save the generated embeddings (default: data/processed/embeddings.npy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding generation (default: 256)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from bigrag.data.embedder import generate_embeddings

    generate_embeddings(
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
