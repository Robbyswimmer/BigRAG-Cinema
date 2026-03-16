#!/usr/bin/env python3
"""Generate LaTeX tables from benchmark results.

Usage:
    python scripts/generate_tables.py [--metrics-dir results/raw_metrics] [--output-dir results/tables]
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from benchmark results.")
    parser.add_argument(
        "--metrics-dir", type=str, default="results/raw_metrics",
        help="Directory containing raw benchmark metrics (default: results/raw_metrics)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/tables",
        help="Directory to save generated tables (default: results/tables)",
    )
    parser.add_argument(
        "--recall-results", type=str, default=None,
        help="Path to recall_results.json (default: results/recall_results.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from bigrag.analysis.report_tables import generate_all_tables
    generate_all_tables(
        args.metrics_dir,
        args.output_dir,
        recall_results_path=args.recall_results,
    )
