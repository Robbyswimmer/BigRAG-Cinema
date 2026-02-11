"""
bigrag.benchmark.workload_generator -- Synthetic query workload generation.

Generates a reproducible set of queries with varying filter
selectivities and query texts for benchmarking purposes.
"""

from __future__ import annotations


def generate_workload(num_queries: int = 1000, seed: int = 42) -> list[dict]:
    """Generate a list of benchmark query specifications.

    Parameters
    ----------
    num_queries : int
        Number of queries to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Each dict contains keys such as ``"text"``, ``"time_range"``,
        ``"score_range"``, ``"user_ids"``, and ``"selectivity_bucket"``.
    """
    raise NotImplementedError("generate_workload is not yet implemented")
