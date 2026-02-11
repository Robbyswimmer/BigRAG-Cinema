"""
bigrag.engine.query_interface -- Top-level query API.

Provides a single entry-point function that accepts a natural-language
query string, optional metadata filters, and a strategy name, then
delegates to the matching ExecutionStrategy to produce results.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def execute_query(
    spark: "SparkSession",
    text: str,
    filters: Optional[dict] = None,
    strategy_name: str = "adaptive",
    top_k: int = 10,
) -> "DataFrame":
    """Run a combined vector + metadata query and return ranked results.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    text : str
        Natural-language query to embed and search for.
    filters : dict | None
        Optional metadata filter specification (time_range, score_range,
        user_ids).
    strategy_name : str
        Name of the execution strategy (see ``strategies.registry``).
    top_k : int
        Number of results to return.

    Returns
    -------
    pyspark.sql.DataFrame
        Top-K results sorted by relevance.
    """
    raise NotImplementedError("execute_query is not yet implemented")
