"""
bigrag.strategies.base -- Abstract base class for execution strategies.

All concrete strategies must subclass ``ExecutionStrategy`` and
implement the ``execute()`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession


class ExecutionStrategy(ABC):
    """Abstract base for hybrid vector + metadata search strategies."""

    @abstractmethod
    def execute(
        self,
        spark: "SparkSession",
        df: "DataFrame",
        query_vec: list[float],
        filters: "Column | None",
        top_k: int,
    ) -> "DataFrame":
        """Execute the strategy and return the top-K results.

        Parameters
        ----------
        spark : SparkSession
            Active Spark session.
        df : pyspark.sql.DataFrame
            Source DataFrame with embeddings.
        query_vec : list[float]
            Query embedding vector.
        filters : pyspark.sql.Column | None
            Pre-built metadata filter expression, or ``None``.
        top_k : int
            Number of results to return.

        Returns
        -------
        pyspark.sql.DataFrame
            Ranked result DataFrame.
        """
        ...
