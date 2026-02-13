#!/usr/bin/env python3
"""Convert raw CSV data and embeddings into Parquet format.

Thin CLI wrapper around bigrag.data.parquet_writer.
"""

import argparse
from pathlib import Path
import sys

# Allow running this script without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Parquet files from raw data and embeddings."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/raw/Reviews.csv",
        help="Path to the raw reviews CSV (default: data/raw/Reviews.csv)",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="data/processed/embeddings.npy",
        help="Path to the embeddings file (default: data/processed/embeddings.npy)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/reviews.parquet",
        help="Path for the output Parquet file (default: data/processed/reviews.parquet)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from bigrag.data.parquet_writer import write_parquet

    write_parquet(
        csv_path=args.csv_path,
        embeddings_path=args.embeddings_path,
        output_path=args.output_path,
    )
