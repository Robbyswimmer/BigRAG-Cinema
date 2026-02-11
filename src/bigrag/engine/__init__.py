"""
bigrag.engine -- Spark-based query engine for vector + metadata search.

Provides SparkSession management, cosine-similarity UDFs, metadata
filter construction, and a high-level query interface that delegates
to pluggable execution strategies.
"""
