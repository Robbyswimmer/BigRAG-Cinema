"""
bigrag.benchmark.metrics_collector -- Per-query and aggregate metrics.

Accumulates timing and quality measurements for individual queries,
then provides summary statistics (mean, median, percentiles, etc.).
"""

from __future__ import annotations


class MetricsCollector:
    """Collect, store, and summarise benchmark measurements."""

    def __init__(self) -> None:
        """Initialise an empty metrics store."""
        self._records: list[dict] = []

    def record(self, query_id: int, metrics: dict) -> None:
        """Record metrics for a single query execution.

        Parameters
        ----------
        query_id : int
            Unique identifier for the query within the workload.
        metrics : dict
            Dictionary of metric name -> value (e.g. latency, recall).
        """
        raise NotImplementedError("MetricsCollector.record is not yet implemented")

    def summarize(self) -> dict:
        """Compute aggregate statistics over all recorded queries.

        Returns
        -------
        dict
            Summary statistics keyed by metric name, each containing
            mean, median, p95, p99, min, max, and count.
        """
        raise NotImplementedError("MetricsCollector.summarize is not yet implemented")
