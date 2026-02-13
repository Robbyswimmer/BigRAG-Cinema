"""
bigrag.engine.vector_search -- Cosine similarity UDF and KNN helpers.

Implements a Spark-compatible cosine similarity function (either as a
UDF or via native array operations) and a top-K retrieval helper that
ranks rows by similarity to a query vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from bigrag.data.schema import COL_EMBEDDING

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
    query_arr = np.asarray(query_vec, dtype=np.float32)
    if query_arr.ndim != 1:
        raise ValueError("query_vec must be a 1-D list of floats")
    query_norm = float(np.linalg.norm(query_arr))

    @F.udf(DoubleType())
    def _cosine(row_embedding):
        if row_embedding is None or query_norm == 0.0:
            return 0.0
        row_arr = np.asarray(row_embedding, dtype=np.float32)
        if row_arr.ndim != 1 or row_arr.shape[0] != query_arr.shape[0]:
            return 0.0
        row_norm = float(np.linalg.norm(row_arr))
        if row_norm == 0.0:
            return 0.0
        return float(np.dot(query_arr, row_arr) / (query_norm * row_norm))

    return _cosine(F.col(col_name))


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
    if k <= 0:
        raise ValueError("k must be positive")
    with_score = df.withColumn("similarity", cosine_similarity_udf(query_vec, COL_EMBEDDING))
    return with_score.orderBy(F.col("similarity").desc()).limit(int(k))
