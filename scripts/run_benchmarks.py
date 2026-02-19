#!/usr/bin/env python3
"""Run benchmark experiments across all query strategies.

Thin CLI wrapper around bigrag.benchmark.experiment_runner.
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
        description="Run benchmark experiments for all query strategies."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/experiment_config.yaml",
        help="Path to benchmark configuration file (default: conf/experiment_config.yaml)",
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
    parser.add_argument(
        "--num-reps",
        type=int,
        default=None,
        help="Number of repetitions per strategy/fraction (default: from config)",
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
        num_repetitions=args.num_reps,
    )
