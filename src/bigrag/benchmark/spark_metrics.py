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

    # Capture shuffle bytes from the Spark listener accumulator
    try:
        status = sc.statusTracker()
        shuffle_read = 0
        shuffle_write = 0
        total_tasks = 0
        executor_run_time_ms = 0
        for job_id in (status.getActiveJobIds() or []):
            job_info = status.getJobInfo(job_id)
            if job_info:
                for stage_id in (job_info.stageIds or []):
                    stage_info = status.getStageInfo(stage_id)
                    if stage_info:
                        total_tasks += stage_info.numActiveTasks + stage_info.numCompletedTasks
        metrics["total_tasks"] = total_tasks
    except Exception:
        metrics["total_tasks"] = 0

    # Capture cumulative shuffle metrics from the Spark UI REST API
    try:
        ui_url = sc.uiWebUrl
        if ui_url:
            import json
            import urllib.request
            stages_url = f"{ui_url}/api/v1/applications/{sc.applicationId}/stages"
            with urllib.request.urlopen(stages_url, timeout=5) as resp:
                stages_data = json.loads(resp.read().decode())
            shuffle_read = sum(s.get("shuffleReadBytes", 0) for s in stages_data)
            shuffle_write = sum(s.get("shuffleWriteBytes", 0) for s in stages_data)
            executor_run_time = sum(s.get("executorRunTime", 0) for s in stages_data)
            metrics["shuffle_read_bytes"] = int(shuffle_read)
            metrics["shuffle_write_bytes"] = int(shuffle_write)
            metrics["executor_run_time_ms"] = int(executor_run_time)
    except Exception:
        metrics["shuffle_read_bytes"] = 0
        metrics["shuffle_write_bytes"] = 0
        metrics["executor_run_time_ms"] = 0

    return metrics
