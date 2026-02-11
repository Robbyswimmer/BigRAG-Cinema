"""
bigrag.engine.vector_search -- Cosine similarity UDF and KNN helpers.

Implements a Spark-compatible cosine similarity function (either as a
UDF or via native array operations) and a top-K retrieval helper that
ranks rows by similarity to a query vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def cosine_similarity_udf(query_vec: list[float], col_name: str = "embedding"):
    """Return a Spark Column expression computing cosine similarity.

    Parameters
    ----------
    query_vec : list[float]
        The query embedding vector.
    col_name : str
        Name of the DataFrame column holding row embeddings.

    Returns
    -------
    pyspark.sql.Column
        A column expression that evaluates to the cosine similarity
        between *query_vec* and each row's embedding.
    """
    raise NotImplementedError("cosine_similarity_udf is not yet implemented")


def top_k_similar(df: "DataFrame", query_vec: list[float], k: int = 10) -> "DataFrame":
    """Return the *k* most similar rows in *df* to *query_vec*.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame containing an ``embedding`` column.
    query_vec : list[float]
        The query embedding vector.
    k : int
        Number of nearest neighbours to return.

    Returns
    -------
    pyspark.sql.DataFrame
        Top-K rows sorted by descending cosine similarity.
    """
    raise NotImplementedError("top_k_similar is not yet implemented")
