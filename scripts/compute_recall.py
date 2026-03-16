#!/usr/bin/env python3
"""Compute Recall@K for each strategy using filter_first as ground truth.

filter_first applies metadata filters then performs exact brute-force cosine
search on the full filtered set, making it the ground-truth retrieval.
Other strategies may miss relevant results due to truncation or merging.

Usage:
    python scripts/compute_recall.py [--data-dir data/parquet] \
        [--num-queries 15] [--top-k 10] [--output results/recall_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Recall@K for each strategy.")
    parser.add_argument(
        "--data-dir", type=str, default="data/parquet",
        help="Directory containing parquet dataset subdirectories (default: data/parquet)",
    )
    parser.add_argument(
        "--num-queries", type=int, default=15,
        help="Number of queries to evaluate (default: 15)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-K parameter for retrieval (default: 10)",
    )
    parser.add_argument(
        "--output", type=str, default="results/recall_results.json",
        help="Output path for recall results JSON (default: results/recall_results.json)",
    )
    parser.add_argument(
        "--profile", type=str, default="conf/cluster_profiles/local.yaml",
        help="Spark cluster profile YAML (default: conf/cluster_profiles/local.yaml)",
    )
    return parser.parse_args()


def _row_key(row) -> tuple:
    """Composite key for row identity (matches hybrid_parallel dedup key)."""
    return (
        row["asin"],
        row["user_id"],
        row["timestamp"],
        row["text"],
    )


def compute_recall_for_dataset(
    spark,
    parquet_path: Path,
    dataset_name: str,
    num_queries: int,
    top_k: int,
) -> dict:
    """Compute Recall@K for all strategies on a single dataset."""
    from bigrag.benchmark.workload_generator import generate_workload
    from bigrag.data.embedder import generate_embeddings
    from bigrag.engine.metadata_filter import build_filters
    from bigrag.strategies.registry import get_strategy

    strategy_names = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]

    df = spark.read.parquet(str(parquet_path))
    total_rows = df.count()
    print(f"\n  {dataset_name}: {total_rows:,} rows")

    workload = generate_workload(num_queries=num_queries, seed=42)

    # Per-strategy, per-selectivity recall tracking
    recall_by_strategy: dict[str, list[float]] = {s: [] for s in strategy_names}
    recall_by_bucket: dict[str, dict[str, list[float]]] = {
        s: {"low": [], "medium": [], "high": []} for s in strategy_names
    }

    ground_truth_strategy = get_strategy("filter_first")

    for qi, query in enumerate(workload):
        query_vec = generate_embeddings(texts=[query["text"]])[0].tolist()
        filters = build_filters(
            time_range=query.get("time_range"),
            score_range=query.get("score_range"),
            user_ids=query.get("user_ids"),
        )
        bucket = query.get("selectivity_bucket", "high")

        # Ground truth: filter_first results
        gt_result = ground_truth_strategy.execute(
            spark=spark, df=df, query_vec=query_vec, filters=filters, top_k=top_k,
        )
        gt_rows = gt_result.collect()
        gt_keys = {_row_key(r) for r in gt_rows}

        if not gt_keys:
            # No ground truth results — skip this query
            for s in strategy_names:
                recall_by_strategy[s].append(1.0)
                recall_by_bucket[s][bucket].append(1.0)
            continue

        for strategy_name in strategy_names:
            if strategy_name == "filter_first":
                recall = 1.0
            else:
                strategy_obj = get_strategy(strategy_name)
                strat_result = strategy_obj.execute(
                    spark=spark, df=df, query_vec=query_vec,
                    filters=filters, top_k=top_k,
                )
                strat_rows = strat_result.collect()
                strat_keys = {_row_key(r) for r in strat_rows}
                overlap = len(gt_keys & strat_keys)
                recall = overlap / len(gt_keys)

            recall_by_strategy[strategy_name].append(recall)
            recall_by_bucket[strategy_name][bucket].append(recall)

        print(f"    Query {qi+1}/{num_queries} done", end="\r")

    print(f"    Completed {num_queries} queries" + " " * 20)

    # Summarize
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    summary = {}
    for s in strategy_names:
        summary[s] = {
            "mean_recall": round(_mean(recall_by_strategy[s]), 4),
            "min_recall": round(min(recall_by_strategy[s]), 4) if recall_by_strategy[s] else 0.0,
            "max_recall": round(max(recall_by_strategy[s]), 4) if recall_by_strategy[s] else 0.0,
            "num_queries": len(recall_by_strategy[s]),
            "by_selectivity": {
                bucket: round(_mean(recall_by_bucket[s][bucket]), 4)
                for bucket in ["low", "medium", "high"]
                if recall_by_bucket[s][bucket]
            },
        }

    return {
        "dataset": dataset_name,
        "total_rows": total_rows,
        "num_queries": num_queries,
        "top_k": top_k,
        "strategies": summary,
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir = data_dir.resolve()

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # Auto-discover parquet datasets
    datasets: list[tuple[str, Path]] = []
    if data_dir.is_dir():
        for p in sorted(data_dir.iterdir()):
            if p.is_dir() and any(p.glob("*.parquet")):
                datasets.append((p.name, p))
            elif p.suffix == ".parquet":
                datasets.append((p.stem, p))

    if not datasets:
        print(f"No parquet datasets found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s): {', '.join(d[0] for d in datasets)}")

    from bigrag.engine.spark_session import create_spark_session
    spark = create_spark_session(profile_path)

    all_results = []
    try:
        for ds_name, ds_path in datasets:
            result = compute_recall_for_dataset(
                spark=spark,
                parquet_path=ds_path,
                dataset_name=ds_name,
                num_queries=args.num_queries,
                top_k=args.top_k,
            )
            all_results.append(result)
    finally:
        spark.stop()

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRecall results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Recall@K Summary':^80}")
    print("=" * 80)
    header = f"{'Dataset':<25} {'Filter-First':>13} {'Vector-First':>13} {'Hybrid':>13} {'Adaptive':>13}"
    print(header)
    print("-" * 80)
    for result in all_results:
        strats = result["strategies"]
        row = f"{result['dataset']:<25}"
        for s in ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]:
            row += f" {strats[s]['mean_recall']:>12.4f}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
