"""
bigrag.strategies.vector_first -- Vector-search-then-filter strategy.

Performs cosine-similarity ranking on the full dataset first, then
applies metadata filters to the top candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bigrag.strategies.base import ExecutionStrategy

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession


class VectorFirstStrategy(ExecutionStrategy):
    """Perform vector similarity search before metadata filtering."""

    def execute(
        self,
        spark: "SparkSession",
        df: "DataFrame",
        query_vec: list[float],
        filters: "Column | None",
        top_k: int,
    ) -> "DataFrame":
        """Rank by cosine similarity, then apply metadata filters."""
        raise NotImplementedError("VectorFirstStrategy.execute is not yet implemented")
