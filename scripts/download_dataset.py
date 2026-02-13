#!/usr/bin/env python3
"""Download the Amazon Reviews 2023 dataset.

Thin CLI wrapper around bigrag.data.downloader.
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
        description="Download the Amazon Reviews 2023 dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save the downloaded dataset (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from bigrag.data.downloader import download

    download(output_dir=args.output_dir, force=args.force)
