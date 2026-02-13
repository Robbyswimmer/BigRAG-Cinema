"""
bigrag.strategies.filter_first -- Filter-then-vector-search strategy.

Applies metadata filters first to reduce the candidate set, then
performs cosine-similarity ranking on the filtered rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bigrag.engine.vector_search import top_k_similar
from bigrag.strategies.base import ExecutionStrategy

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession


class FilterFirstStrategy(ExecutionStrategy):
    """Apply metadata filters before vector similarity search."""

    def execute(
        self,
        spark: "SparkSession",
        df: "DataFrame",
        query_vec: list[float],
        filters: "Column | None",
        top_k: int,
    ) -> "DataFrame":
        """Filter the DataFrame, then rank by cosine similarity."""
        candidate_df = df.filter(filters) if filters is not None else df
        return top_k_similar(candidate_df, query_vec=query_vec, k=top_k)
