"""
bigrag.strategies.adaptive -- Heuristic strategy selection at runtime.

Inspects query characteristics (filter selectivity estimate, dataset
size, etc.) and delegates to the most appropriate concrete strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bigrag.strategies.base import ExecutionStrategy

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession


class AdaptiveStrategy(ExecutionStrategy):
    """Select and delegate to the best strategy based on query heuristics."""

    def execute(
        self,
        spark: "SparkSession",
        df: "DataFrame",
        query_vec: list[float],
        filters: "Column | None",
        top_k: int,
    ) -> "DataFrame":
        """Analyse the query, pick a strategy, and delegate execution."""
        raise NotImplementedError("AdaptiveStrategy.execute is not yet implemented")
