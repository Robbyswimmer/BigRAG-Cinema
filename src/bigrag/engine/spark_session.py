"""
bigrag.engine.spark_session -- SparkSession factory from a cluster profile.

Reads a YAML configuration file that describes cluster resources
(executor memory, cores, etc.) and returns a ready-to-use SparkSession.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
    raise NotImplementedError("create_spark_session is not yet implemented")
