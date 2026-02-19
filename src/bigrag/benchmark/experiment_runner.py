"""
bigrag.benchmark.experiment_runner -- End-to-end experiment orchestration.

Loads data, warms up the cluster, runs the workload against each
strategy / data fraction, collects metrics, and saves results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from bigrag.benchmark.metrics_collector import MetricsCollector
from bigrag.benchmark.spark_metrics import collect_spark_metrics
from bigrag.benchmark.timer import Timer
from bigrag.benchmark.workload_generator import generate_workload
from bigrag.data.embedder import generate_embeddings
from bigrag.engine.metadata_filter import build_filters
from bigrag.engine.spark_session import create_spark_session
from bigrag.strategies.registry import STRATEGY_REGISTRY, get_strategy
from pyspark import StorageLevel
from bigrag.utils.config import load_config
from bigrag.utils.io import save_json

STRATEGY_ALIASES: dict[str, str] = {
    "brute_force_cosine": "vector_first",
    "broadcast_join": "hybrid_parallel",
    "partitioned_search": "filter_first",
    "lsh_approx": "adaptive",
}


def _normalize_strategy_names(raw: list[Any]) -> list[str]:
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            candidate = str(item.get("name", "")).strip()
        else:
            candidate = str(item).strip()
        if not candidate:
            continue
        key = candidate.lower()
        names.append(STRATEGY_ALIASES.get(key, key))
    return names


def _resolve_output_path(config: dict) -> Path:
    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("results_dir", "results/raw_metrics")
    return Path(output_dir).expanduser().resolve()


def _resolve_source_path(config: dict) -> Path:
    dataset_cfg = config.get("dataset", {})
    source_path = dataset_cfg.get("source_path", "data/processed/reviews.parquet")
    return Path(source_path).expanduser().resolve()


def _detect_fraction_anomalies(summary: dict) -> list[str]:
    anomalies: list[str] = []
    metric_summary = summary.get("metrics", {})
    latency = metric_summary.get("latency_ms")
    throughput = metric_summary.get("throughput_qps")
    results = metric_summary.get("result_count")

    if latency:
        median = float(latency.get("median", 0.0))
        p99 = float(latency.get("p99", 0.0))
        if median > 0 and (p99 / median) > 10.0:
            anomalies.append("high_latency_tail")
        if median > 5000:
            anomalies.append("very_high_median_latency")

    if throughput and float(throughput.get("mean", 0.0)) < 0.1:
        anomalies.append("very_low_throughput")

    if results and float(results.get("mean", 0.0)) <= 0.0:
        anomalies.append("zero_result_queries")

    return anomalies


def run_experiment(config: dict) -> dict:
    """Execute a full benchmark experiment defined by *config*.

    Parameters
    ----------
    config : dict
        Experiment configuration containing keys such as
        ``"data_path"``, ``"strategies"``, ``"fractions"``,
        ``"num_queries"``, ``"warmup_rounds"``, and ``"output_dir"``.

    Returns
    -------
    dict
        Nested results structure keyed by strategy name and fraction,
        containing raw and summary metrics.
    """
    profile_path = Path(config.get("profile_path", "conf/cluster_profiles/local.yaml"))
    source_path = _resolve_source_path(config)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {source_path}")

    fractions = config.get("dataset", {}).get("fractions", [1.0])
    query_cfg = config.get("queries", {})
    num_queries = int(query_cfg.get("num_queries", 20))
    seed = int(query_cfg.get("seed", 42))
    warmup_rounds = int(config.get("misc", {}).get("warm_up_runs", 1))
    num_repetitions = int(config.get("misc", {}).get("num_repetitions", 1))
    default_top_k = 10

    requested_names = _normalize_strategy_names(config.get("strategies", []))
    if not requested_names:
        requested_names = sorted(STRATEGY_REGISTRY.keys())

    supported_names = [s for s in requested_names if s in STRATEGY_REGISTRY]
    skipped_names = [s for s in requested_names if s not in STRATEGY_REGISTRY]
    if not supported_names:
        supported_names = ["adaptive"]

    workload = generate_workload(num_queries=num_queries, seed=seed)

    spark = create_spark_session(profile_path)
    try:
        full_df = spark.read.parquet(str(source_path))
        total_rows = full_df.count()

        results: dict[str, Any] = {
            "config": {
                "profile_path": str(profile_path),
                "source_path": str(source_path),
                "fractions": fractions,
                "num_queries": num_queries,
                "seed": seed,
                "warmup_rounds": warmup_rounds,
                "num_repetitions": num_repetitions,
                "supported_strategies": supported_names,
                "skipped_strategies": skipped_names,
                "total_rows": total_rows,
            },
            "strategies": {},
        }

        total_measured = (
            len(supported_names) * len(fractions) * num_repetitions * num_queries
        )
        pbar = tqdm(total=total_measured, desc="Benchmark", unit="query")

        for strategy_name in supported_names:
            strategy_obj = get_strategy(strategy_name)
            strategy_entry: dict[str, Any] = {"fractions": {}}
            for frac in fractions:
                collector = MetricsCollector()
                frac_value = float(frac)
                frac_rows = max(1, int(total_rows * frac_value))

                pbar.set_description(
                    f"{strategy_name} | {frac_value:.0%} ({frac_rows:,} rows)"
                )

                df = full_df.limit(frac_rows).persist(StorageLevel.MEMORY_AND_DISK)
                df.count()

                for repetition_id in range(num_repetitions):
                    # Warm-up runs are not measured.
                    for i in range(min(warmup_rounds, len(workload))):
                        q = workload[i]
                        query_vec = generate_embeddings(texts=[q["text"]])[0].tolist()
                        filters = build_filters(
                            time_range=q.get("time_range"),
                            score_range=q.get("score_range"),
                            user_ids=q.get("user_ids"),
                        )
                        _ = strategy_obj.execute(
                            spark=spark,
                            df=df,
                            query_vec=query_vec,
                            filters=filters,
                            top_k=default_top_k,
                        ).count()

                    for query in workload:
                        query_id = int(query["query_id"])
                        query_vec = generate_embeddings(texts=[query["text"]])[0].tolist()
                        filters = build_filters(
                            time_range=query.get("time_range"),
                            score_range=query.get("score_range"),
                            user_ids=query.get("user_ids"),
                        )
                        with Timer() as timer:
                            out = strategy_obj.execute(
                                spark=spark,
                                df=df,
                                query_vec=query_vec,
                                filters=filters,
                                top_k=default_top_k,
                            )
                            result_count = out.count()

                        latency_ms = timer.elapsed_s * 1000.0
                        throughput_qps = (
                            0.0 if timer.elapsed_s <= 0 else 1.0 / timer.elapsed_s
                        )
                        spark_metrics = collect_spark_metrics(spark)
                        collector.record(
                            query_id=query_id,
                            metrics={
                                "fraction": frac_value,
                                "repetition_id": float(repetition_id),
                                "latency_ms": latency_ms,
                                "throughput_qps": throughput_qps,
                                "result_count": float(result_count),
                                "active_stage_count": float(
                                    spark_metrics.get("active_stage_count", 0)
                                ),
                            },
                        )
                        pbar.update(1)

                fraction_key = f"{frac_value:.4f}"
                fraction_summary = collector.summarize()
                strategy_entry["fractions"][fraction_key] = {
                    "records": collector.records,
                    "summary": fraction_summary,
                    "anomalies": _detect_fraction_anomalies(fraction_summary),
                }
                df.unpersist()

            strategy_anomalies: list[str] = []
            for fraction_payload in strategy_entry["fractions"].values():
                for item in fraction_payload.get("anomalies", []):
                    if item not in strategy_anomalies:
                        strategy_anomalies.append(item)
            strategy_entry["validation"] = {
                "status": "warn" if strategy_anomalies else "ok",
                "anomalies": strategy_anomalies,
            }
            results["strategies"][strategy_name] = strategy_entry

        pbar.set_description("Benchmark complete")
        pbar.close()

        expected_records_per_fraction = num_queries * num_repetitions
        completeness_issues: list[str] = []
        for strategy_name, strategy_payload in results["strategies"].items():
            for fraction_key, fraction_payload in strategy_payload["fractions"].items():
                actual_count = int(fraction_payload["summary"].get("count", 0))
                if actual_count != expected_records_per_fraction:
                    completeness_issues.append(
                        f"{strategy_name}:{fraction_key}:expected="
                        f"{expected_records_per_fraction},actual={actual_count}"
                    )
        results["validation"] = {
            "status": "warn" if completeness_issues else "ok",
            "expected_records_per_fraction": expected_records_per_fraction,
            "completeness_issues": completeness_issues,
        }

        output_dir = _resolve_output_path(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(results, output_dir / "benchmark_results.json")
        return results
    finally:
        spark.stop()


def run_experiments(
    config_path: str = "conf/experiment_config.yaml",
    output_dir: str | None = None,
    strategies: list[str] | None = None,
    num_queries: int | None = None,
    num_repetitions: int | None = None,
) -> dict:
    """CLI-facing wrapper used by scripts/run_benchmarks.py."""
    config = load_config(Path(config_path))
    if output_dir is not None:
        config.setdefault("output", {})
        config["output"]["results_dir"] = output_dir
    if strategies is not None:
        config["strategies"] = strategies
    if num_queries is not None:
        config.setdefault("queries", {})
        config["queries"]["num_queries"] = int(num_queries)
    if num_repetitions is not None:
        config.setdefault("misc", {})
        config["misc"]["num_repetitions"] = int(num_repetitions)
    return run_experiment(config)
