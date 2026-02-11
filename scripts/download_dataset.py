#!/usr/bin/env python3
"""Download the Amazon Reviews 2023 dataset.

Thin CLI wrapper around bigrag.data.downloader.
"""

import argparse
import sys


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
