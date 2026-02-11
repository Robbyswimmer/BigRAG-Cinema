"""
bigrag.strategies.hybrid_parallel -- Concurrent filter + vector merge.

Runs metadata filtering and vector similarity search in parallel,
then merges the two result sets to produce the final ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bigrag.strategies.base import ExecutionStrategy

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession


class HybridParallelStrategy(ExecutionStrategy):
    """Run filter and vector search concurrently, then merge results."""

    def execute(
        self,
        spark: "SparkSession",
        df: "DataFrame",
        query_vec: list[float],
        filters: "Column | None",
        top_k: int,
    ) -> "DataFrame":
        """Execute filter and vector branches in parallel and merge."""
        raise NotImplementedError(
            "HybridParallelStrategy.execute is not yet implemented"
        )
