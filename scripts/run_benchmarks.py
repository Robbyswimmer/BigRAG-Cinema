#!/usr/bin/env python3
"""Run benchmark experiments across all query strategies.

Thin CLI wrapper around bigrag.benchmark.experiment_runner.
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments for all query strategies."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to benchmark configuration file (default: configs/benchmark.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/raw_metrics",
        help="Directory to save raw benchmark metrics (default: results/raw_metrics)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Specific strategies to benchmark (default: all)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries to run (default: from config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from bigrag.benchmark.experiment_runner import run_experiments

    run_experiments(
        config_path=args.config,
        output_dir=args.output_dir,
        strategies=args.strategies,
        num_queries=args.num_queries,
    )
