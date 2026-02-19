"""
bigrag.benchmark.workload_generator -- Synthetic query workload generation.

Generates a reproducible set of queries with varying filter
selectivities and query texts for benchmarking purposes.
"""

from __future__ import annotations

import random


QUERY_TEMPLATES = [
    "excellent product works great",
    "terrible quality broke after one use",
    "good value for the price",
    "amazing gift for birthday",
    "disappointing not as described",
    "perfect exactly what I needed",
    "best purchase I have made",
    "waste of money do not buy",
]

USER_POOL = [f"user_{i:04d}" for i in range(1, 501)]
SELECTIVITY_BUCKETS = ("low", "medium", "high")


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
    if num_queries <= 0:
        return []

    rng = random.Random(seed)
    workload: list[dict] = []
    for i in range(num_queries):
        bucket = SELECTIVITY_BUCKETS[i % len(SELECTIVITY_BUCKETS)]
        text = rng.choice(QUERY_TEMPLATES)

        if bucket == "low":
            score_range = (4.5, 5.0)
            user_ids = rng.sample(USER_POOL, k=2)
        elif bucket == "medium":
            score_range = (3.0, 5.0)
            user_ids = rng.sample(USER_POOL, k=10)
        else:
            score_range = (1.0, 5.0)
            user_ids = None

        start_year = rng.randint(2008, 2018)
        end_year = min(start_year + rng.randint(1, 4), 2023)
        time_range = (f"{start_year}-01-01T00:00:00Z", f"{end_year}-12-31T23:59:59Z")

        workload.append(
            {
                "query_id": i,
                "text": text,
                "time_range": time_range,
                "score_range": score_range,
                "user_ids": user_ids,
                "selectivity_bucket": bucket,
            }
        )
    return workload
