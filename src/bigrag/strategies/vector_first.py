"""
bigrag.strategies.vector_first -- Vector-search-then-filter strategy.

Performs cosine-similarity ranking on the full dataset first, then
applies metadata filters to the top candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from bigrag.engine.vector_search import top_k_similar
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
        pre_k = max(top_k * 10, 100)
        vector_ranked = top_k_similar(df, query_vec=query_vec, k=pre_k)
        if filters is not None:
            vector_ranked = vector_ranked.filter(filters)
        return vector_ranked.orderBy(F.col("similarity").desc()).limit(top_k)
