"""
bigrag.benchmark.spark_metrics -- Extract internal Spark execution metrics.

Uses Spark's internal StatusStore via the JVM bridge to capture shuffle
bytes read/written, number of tasks, stage durations, and other
executor-level statistics after a job completes.  This approach works
regardless of whether the Spark UI REST server is enabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class ShuffleTracker:
    """Track cumulative shuffle metrics and compute per-query deltas.

    Usage::

        tracker = ShuffleTracker(spark)
        tracker.snapshot()        # before query
        ... run query ...
        delta = tracker.delta()   # after query
    """

    def __init__(self, spark: "SparkSession") -> None:
        self._spark = spark
        self._prev: dict = {}

    def _read_cumulative(self) -> dict:
        """Read cumulative shuffle metrics from the StatusStore."""
        sc = self._spark.sparkContext
        metrics = {
            "shuffle_read_bytes": 0,
            "shuffle_write_bytes": 0,
            "total_tasks": 0,
            "executor_run_time_ms": 0,
        }

        try:
            # Access Spark's internal StatusStore via JVM bridge.
            # statusStore() retains completed stage metrics in memory.
            jsc = sc._jsc.sc()
            status_store = jsc.statusStore()

            # stageList(statuses) returns completed+active stages
            from py4j.java_collections import ListConverter
            stage_list = status_store.stageList(None)

            for i in range(stage_list.size()):
                stage_data = stage_list.apply(i)
                try:
                    sm = stage_data.shuffleReadBytes()
                    metrics["shuffle_read_bytes"] += int(sm)
                except Exception:
                    pass
                try:
                    sm = stage_data.shuffleWriteBytes()
                    metrics["shuffle_write_bytes"] += int(sm)
                except Exception:
                    pass
                try:
                    metrics["total_tasks"] += int(stage_data.numCompleteTasks())
                except Exception:
                    pass
                try:
                    metrics["executor_run_time_ms"] += int(stage_data.executorRunTime())
                except Exception:
                    pass
        except Exception:
            # Fallback: try the REST API approach
            try:
                ui_url = sc.uiWebUrl
                if ui_url:
                    import json
                    import urllib.request
                    stages_url = f"{ui_url}/api/v1/applications/{sc.applicationId}/stages"
                    with urllib.request.urlopen(stages_url, timeout=5) as resp:
                        stages_data = json.loads(resp.read().decode())
                    metrics["shuffle_read_bytes"] = sum(
                        s.get("shuffleReadBytes", 0) for s in stages_data
                    )
                    metrics["shuffle_write_bytes"] = sum(
                        s.get("shuffleWriteBytes", 0) for s in stages_data
                    )
                    metrics["total_tasks"] = sum(
                        s.get("numCompleteTasks", 0) for s in stages_data
                    )
                    metrics["executor_run_time_ms"] = sum(
                        s.get("executorRunTime", 0) for s in stages_data
                    )
            except Exception:
                pass

        return metrics

    def snapshot(self) -> None:
        """Capture cumulative metrics before a query."""
        self._prev = self._read_cumulative()

    def delta(self) -> dict:
        """Compute delta metrics since the last snapshot."""
        current = self._read_cumulative()
        result = {}
        for key in current:
            result[key] = current[key] - self._prev.get(key, 0)
        self._prev = current
        return result


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
        and ``"executor_run_time_ms"``.
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

    # Use StatusStore for shuffle metrics (works even without UI server)
    try:
        jsc = sc._jsc.sc()
        status_store = jsc.statusStore()
        stage_list = status_store.stageList(None)

        shuffle_read = 0
        shuffle_write = 0
        total_tasks = 0
        executor_run_time = 0

        for i in range(stage_list.size()):
            stage_data = stage_list.apply(i)
            try:
                shuffle_read += int(stage_data.shuffleReadBytes())
            except Exception:
                pass
            try:
                shuffle_write += int(stage_data.shuffleWriteBytes())
            except Exception:
                pass
            try:
                total_tasks += int(stage_data.numCompleteTasks())
            except Exception:
                pass
            try:
                executor_run_time += int(stage_data.executorRunTime())
            except Exception:
                pass

        metrics["shuffle_read_bytes"] = shuffle_read
        metrics["shuffle_write_bytes"] = shuffle_write
        metrics["total_tasks"] = total_tasks
        metrics["executor_run_time_ms"] = executor_run_time
    except Exception:
        # Fallback: REST API if StatusStore is unavailable
        try:
            ui_url = sc.uiWebUrl
            if ui_url:
                import json
                import urllib.request
                stages_url = f"{ui_url}/api/v1/applications/{sc.applicationId}/stages"
                with urllib.request.urlopen(stages_url, timeout=5) as resp:
                    stages_data = json.loads(resp.read().decode())
                metrics["shuffle_read_bytes"] = sum(
                    s.get("shuffleReadBytes", 0) for s in stages_data
                )
                metrics["shuffle_write_bytes"] = sum(
                    s.get("shuffleWriteBytes", 0) for s in stages_data
                )
                metrics["total_tasks"] = sum(
                    s.get("numCompleteTasks", 0) for s in stages_data
                )
                metrics["executor_run_time_ms"] = sum(
                    s.get("executorRunTime", 0) for s in stages_data
                )
            else:
                metrics["shuffle_read_bytes"] = 0
                metrics["shuffle_write_bytes"] = 0
                metrics["total_tasks"] = 0
                metrics["executor_run_time_ms"] = 0
        except Exception:
            metrics["shuffle_read_bytes"] = 0
            metrics["shuffle_write_bytes"] = 0
            metrics["total_tasks"] = 0
            metrics["executor_run_time_ms"] = 0

    return metrics
