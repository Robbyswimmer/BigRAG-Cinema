#!/usr/bin/env python3
"""Download the full Amazon Reviews 2023 dataset from Hugging Face.

Downloads all 34 review category JSONL files.
Usage:
    python scripts/download_hf_data.py              # all categories
    python scripts/download_hf_data.py --only Digital_Music All_Beauty Video_Games
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

ALL_CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Unknown",
    "Video_Games",
]

LOCAL_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 from HF")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Download only these categories (e.g. --only Digital_Music Video_Games)",
    )
    args = parser.parse_args()

    categories = args.only if args.only else ALL_CATEGORIES
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    total = len(categories)
    for i, cat in enumerate(categories, 1):
        dest = LOCAL_DIR / "raw" / "review_categories" / f"{cat}.jsonl"
        if dest.exists():
            size = dest.stat().st_size / (1024 * 1024)
            print(f"[{i}/{total}] {cat} already exists ({size:.1f} MB) — skipping")
            continue

        print(f"[{i}/{total}] Downloading {cat}...")
        hf_hub_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            filename=f"raw/review_categories/{cat}.jsonl",
            repo_type="dataset",
            local_dir=str(LOCAL_DIR),
        )
        size = dest.stat().st_size / (1024 * 1024) if dest.exists() else 0
        print(f"  Done: {size:.1f} MB")

    print("\nAll downloads complete.")
    for cat in categories:
        p = LOCAL_DIR / "raw" / "review_categories" / f"{cat}.jsonl"
        size = p.stat().st_size / (1024 * 1024) if p.exists() else 0
        print(f"  {cat}: {size:.1f} MB")


if __name__ == "__main__":
    main()
