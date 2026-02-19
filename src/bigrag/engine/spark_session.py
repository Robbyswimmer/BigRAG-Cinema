"""
bigrag.engine.spark_session -- SparkSession factory from a cluster profile.

Reads a YAML configuration file that describes cluster resources
(executor memory, cores, etc.) and returns a ready-to-use SparkSession.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bigrag.utils.config import load_config

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def create_spark_session(profile_path: Path) -> "SparkSession":
    """Build and return a SparkSession configured from *profile_path*.

    Parameters
    ----------
    profile_path : Path
        Path to a YAML file containing Spark configuration key-value
        pairs (e.g. ``spark.executor.memory``, ``spark.executor.cores``).

    Returns
    -------
    SparkSession
        A configured SparkSession instance.
    """
    from pyspark.sql import SparkSession

    profile = load_config(profile_path)
    spark_cfg = profile.get("spark", {})

    app_name = spark_cfg.get("app_name", "BigRAG-Cinema")
    master = spark_cfg.get("master", "local[*]")

    builder = SparkSession.builder.appName(app_name).master(master)

    deploy_mode = spark_cfg.get("deploy_mode")
    if deploy_mode:
        builder = builder.config("spark.submit.deployMode", deploy_mode)

    driver = spark_cfg.get("driver", {})
    if "memory" in driver:
        builder = builder.config("spark.driver.memory", str(driver["memory"]))
    if "cores" in driver:
        builder = builder.config("spark.driver.cores", str(driver["cores"]))
    if "max_result_size" in driver:
        builder = builder.config(
            "spark.driver.maxResultSize", str(driver["max_result_size"])
        )

    executor = spark_cfg.get("executor", {})
    if "memory" in executor:
        builder = builder.config("spark.executor.memory", str(executor["memory"]))
    if "cores" in executor:
        builder = builder.config("spark.executor.cores", str(executor["cores"]))
    if "instances" in executor:
        builder = builder.config("spark.executor.instances", str(executor["instances"]))

    sql_cfg = spark_cfg.get("sql", {})
    if "shuffle_partitions" in sql_cfg:
        builder = builder.config(
            "spark.sql.shuffle.partitions", str(sql_cfg["shuffle_partitions"])
        )
    if "adaptive_enabled" in sql_cfg:
        builder = builder.config(
            "spark.sql.adaptive.enabled",
            str(bool(sql_cfg["adaptive_enabled"])).lower(),
        )
    if "adaptive_coalesce_partitions" in sql_cfg:
        builder = builder.config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            str(bool(sql_cfg["adaptive_coalesce_partitions"])).lower(),
        )
    if "parquet_compression" in sql_cfg:
        builder = builder.config(
            "spark.sql.parquet.compression.codec", str(sql_cfg["parquet_compression"])
        )
    if "arrow_enabled" in sql_cfg:
        builder = builder.config(
            "spark.sql.execution.arrow.pyspark.enabled",
            str(bool(sql_cfg["arrow_enabled"])).lower(),
        )

    mem_cfg = spark_cfg.get("memory", {})
    if "fraction" in mem_cfg:
        builder = builder.config("spark.memory.fraction", str(mem_cfg["fraction"]))
    if "storage_fraction" in mem_cfg:
        builder = builder.config(
            "spark.memory.storageFraction", str(mem_cfg["storage_fraction"])
        )

    if "serializer" in spark_cfg:
        builder = builder.config("spark.serializer", str(spark_cfg["serializer"]))
    if "default_parallelism" in spark_cfg:
        builder = builder.config(
            "spark.default.parallelism", str(spark_cfg["default_parallelism"])
        )

    ui_cfg = spark_cfg.get("ui", {})
    if "enabled" in ui_cfg:
        builder = builder.config("spark.ui.enabled", str(bool(ui_cfg["enabled"])).lower())
    if "port" in ui_cfg:
        builder = builder.config("spark.ui.port", str(ui_cfg["port"]))

    event_log_cfg = spark_cfg.get("event_log", {})
    if "enabled" in event_log_cfg:
        builder = builder.config(
            "spark.eventLog.enabled",
            str(bool(event_log_cfg["enabled"])).lower(),
        )
    if "dir" in event_log_cfg:
        builder = builder.config("spark.eventLog.dir", str(event_log_cfg["dir"]))

    dynamic_cfg = spark_cfg.get("dynamic_allocation", {})
    if dynamic_cfg:
        if "enabled" in dynamic_cfg:
            builder = builder.config(
                "spark.dynamicAllocation.enabled",
                str(bool(dynamic_cfg["enabled"])).lower(),
            )
        if "min_executors" in dynamic_cfg:
            builder = builder.config(
                "spark.dynamicAllocation.minExecutors",
                str(dynamic_cfg["min_executors"]),
            )
        if "max_executors" in dynamic_cfg:
            builder = builder.config(
                "spark.dynamicAllocation.maxExecutors",
                str(dynamic_cfg["max_executors"]),
            )
        if "initial_executors" in dynamic_cfg:
            builder = builder.config(
                "spark.dynamicAllocation.initialExecutors",
                str(dynamic_cfg["initial_executors"]),
            )

    yarn_cfg = profile.get("yarn", {})
    if "queue" in yarn_cfg:
        builder = builder.config("spark.yarn.queue", str(yarn_cfg["queue"]))

    extra_java_options = profile.get("extra_java_options")
    if extra_java_options:
        builder = builder.config("spark.driver.extraJavaOptions", extra_java_options)
        builder = builder.config("spark.executor.extraJavaOptions", extra_java_options)

    return builder.getOrCreate()
