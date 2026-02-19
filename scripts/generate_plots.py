#!/usr/bin/env python3
"""Generate plots and figures from benchmark results.

Thin CLI wrapper around bigrag.analysis.plotting.
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
        description="Generate plots and figures from benchmark results."
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="results/raw_metrics",
        help="Directory containing raw benchmark metrics (default: results/raw_metrics)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to save generated figures (default: results/figures)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output figure format (default: png)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Dataset label for figure titles (e.g. 'Digital_Music')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from bigrag.analysis.plotting import generate_plots

    generate_plots(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        fmt=args.format,
        label=args.label,
    )
