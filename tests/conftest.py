"""Shared test fixtures for BigRAG Cinema tests."""

import os
import pytest

# ---------------------------------------------------------------------------
# Fixture: small SparkSession for local testing
# ---------------------------------------------------------------------------
# @pytest.fixture(scope="session")
# def spark():
#     """Create a minimal local SparkSession for tests."""
#     from pyspark.sql import SparkSession
#     spark = (
#         SparkSession.builder
#         .master("local[2]")
#         .appName("bigrag-tests")
#         .config("spark.sql.shuffle.partitions", "2")
#         .config("spark.ui.enabled", "false")
#         .getOrCreate()
#     )
#     yield spark
#     spark.stop()

# ---------------------------------------------------------------------------
# Fixture: sample reviews DataFrame
# ---------------------------------------------------------------------------
# @pytest.fixture
# def sample_reviews_df(spark):
#     """Load the 5-row sample CSV as a Spark DataFrame."""
#     csv_path = os.path.join(
#         os.path.dirname(__file__), "test_data", "sample_reviews.csv"
#     )
#     return spark.read.csv(csv_path, header=True, inferSchema=True)

# ---------------------------------------------------------------------------
# Fixture: sample embeddings array
# ---------------------------------------------------------------------------
# @pytest.fixture
# def sample_embeddings():
#     """Return a small numpy array simulating embeddings (5 rows x 384 dims)."""
#     import numpy as np
#     rng = np.random.default_rng(42)
#     return rng.standard_normal((5, 384)).astype(np.float32)


@pytest.fixture
def test_data_dir():
    """Return the absolute path to the test_data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")
