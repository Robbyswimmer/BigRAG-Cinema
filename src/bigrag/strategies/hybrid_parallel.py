"""
bigrag.strategies.hybrid_parallel -- Concurrent filter + vector merge.

Runs metadata filtering and vector similarity search in parallel,
then merges the two result sets to produce the final ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from bigrag.data.schema import COL_ASIN, COL_TEXT, COL_TIMESTAMP, COL_USER_ID
from bigrag.engine.vector_search import top_k_similar
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
        branch_k = max(top_k * 5, 50)

        filter_branch_source = df.filter(filters) if filters is not None else df
        filter_branch = top_k_similar(filter_branch_source, query_vec=query_vec, k=branch_k)

        vector_branch = top_k_similar(df, query_vec=query_vec, k=branch_k)
        if filters is not None:
            vector_branch = vector_branch.filter(filters)

        merged = filter_branch.unionByName(vector_branch)
        deduped = merged.dropDuplicates([COL_ASIN, COL_USER_ID, COL_TIMESTAMP, COL_TEXT])
        return deduped.orderBy(F.col("similarity").desc()).limit(top_k)
