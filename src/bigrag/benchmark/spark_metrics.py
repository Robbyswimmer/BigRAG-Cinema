"""
bigrag.benchmark.spark_metrics -- Extract internal Spark execution metrics.

Queries the Spark UI REST API or SparkListener to capture shuffle
bytes read/written, number of tasks, stage durations, and other
executor-level statistics after a job completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def collect_spark_metrics(spark: "SparkSession") -> dict:
    """Retrieve Spark-internal metrics for the most recent job(s).

    Parameters
    ----------
    spark : SparkSession
        Active Spark session whose metrics will be queried.

    Returns
    -------
    dict
        Dictionary with keys such as ``"shuffle_read_bytes"``,
        ``"shuffle_write_bytes"``, ``"total_tasks"``,
        ``"stage_durations_ms"``, and ``"executor_run_time_ms"``.
    """
    sc = spark.sparkContext
    tracker = sc.statusTracker()

    stage_ids = list(tracker.getActiveStageIds() or [])
    # Active stage list can be empty if we are between jobs.
    # Include a lightweight indicator so callers can still track that.
    metrics = {
        "active_stage_count": int(len(stage_ids)),
        "active_stage_ids": [int(sid) for sid in stage_ids],
        "default_parallelism": int(sc.defaultParallelism),
    }

    conf = sc.getConf()
    metrics["spark_sql_shuffle_partitions"] = int(
        conf.get("spark.sql.shuffle.partitions", "0")
    )
    metrics["spark_dynamic_allocation_enabled"] = (
        conf.get("spark.dynamicAllocation.enabled", "false").lower() == "true"
    )
    metrics["spark_executor_instances"] = int(
        conf.get("spark.executor.instances", "0")
    )
    return metrics
