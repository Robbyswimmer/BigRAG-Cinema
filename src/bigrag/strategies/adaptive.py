"""
bigrag.strategies.adaptive -- Heuristic strategy selection at runtime.

Inspects query characteristics (filter selectivity estimate, dataset
size, etc.) and delegates to the most appropriate concrete strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bigrag.strategies.filter_first import FilterFirstStrategy
from bigrag.strategies.hybrid_parallel import HybridParallelStrategy
from bigrag.strategies.base import ExecutionStrategy
from bigrag.strategies.vector_first import VectorFirstStrategy

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession


class AdaptiveStrategy(ExecutionStrategy):
    """Select and delegate to the best strategy based on query heuristics."""

    def _estimate_selectivity(self, df: "DataFrame", filters: "Column | None") -> float:
        if filters is None:
            return 1.0
        sample_df = df.limit(2000)
        sample_total = sample_df.count()
        if sample_total == 0:
            return 1.0
        matched = sample_df.filter(filters).count()
        return float(matched) / float(sample_total)

    def execute(
        self,
        spark: "SparkSession",
        df: "DataFrame",
        query_vec: list[float],
        filters: "Column | None",
        top_k: int,
    ) -> "DataFrame":
        """Analyse the query, pick a strategy, and delegate execution."""
        selectivity = self._estimate_selectivity(df, filters)

        if filters is None:
            strategy: ExecutionStrategy = VectorFirstStrategy()
        elif selectivity <= 0.20:
            strategy = FilterFirstStrategy()
        elif selectivity <= 0.60:
            strategy = HybridParallelStrategy()
        else:
            strategy = VectorFirstStrategy()

        return strategy.execute(
            spark=spark,
            df=df,
            query_vec=query_vec,
            filters=filters,
            top_k=top_k,
        )
