#!/usr/bin/env python3
"""Generate plots and figures from benchmark results.

Thin CLI wrapper around bigrag.analysis.plotting.
"""

import argparse
import sys


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from bigrag.analysis.plotting import generate_plots

    generate_plots(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        fmt=args.format,
    )
