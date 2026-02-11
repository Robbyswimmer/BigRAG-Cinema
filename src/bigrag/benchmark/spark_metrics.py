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
    raise NotImplementedError("collect_spark_metrics is not yet implemented")
