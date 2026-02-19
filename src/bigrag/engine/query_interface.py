"""
bigrag.engine.query_interface -- Top-level query API.

Provides a single entry-point function that accepts a natural-language
query string, optional metadata filters, and a strategy name, then
delegates to the matching ExecutionStrategy to produce results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from bigrag.data.embedder import generate_embeddings
from bigrag.engine.metadata_filter import build_filters
from bigrag.engine.vector_search import top_k_similar
from bigrag.strategies.registry import get_strategy

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

DEFAULT_DATA_PATHS = (
    Path("data/parquet/All_Beauty.parquet"),
    Path("data/parquet/Digital_Music.parquet"),
    Path("data/parquet/Video_Games.parquet"),
    Path("data/processed/reviews.parquet"),
)


def _load_default_df(spark: "SparkSession") -> "DataFrame":
    for path in DEFAULT_DATA_PATHS:
        candidate = path.expanduser().resolve()
        if candidate.exists():
            return spark.read.parquet(str(candidate))
    raise FileNotFoundError(
        "No default parquet dataset found. Checked: "
        + ", ".join(str(p.expanduser().resolve()) for p in DEFAULT_DATA_PATHS)
    )


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
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_vec = generate_embeddings(texts=[text])[0].tolist()
    spec = filters or {}
    filter_expr = build_filters(
        time_range=spec.get("time_range"),
        score_range=spec.get("score_range"),
        user_ids=spec.get("user_ids"),
    )

    df = _load_default_df(spark)

    if strategy_name:
        try:
            strategy = get_strategy(strategy_name)
            return strategy.execute(
                spark=spark,
                df=df,
                query_vec=query_vec,
                filters=filter_expr,
                top_k=top_k,
            )
        except NotImplementedError:
            # Strategy implementations are completed in Phase 3.
            pass

    return top_k_similar(df.filter(filter_expr), query_vec=query_vec, k=top_k)
