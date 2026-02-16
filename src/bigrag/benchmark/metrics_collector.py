"""
bigrag.benchmark.metrics_collector -- Per-query and aggregate metrics.

Accumulates timing and quality measurements for individual queries,
then provides summary statistics (mean, median, percentiles, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary")
        payload = {"query_id": int(query_id), **metrics}
        self._records.append(payload)

    def summarize(self) -> dict:
        """Compute aggregate statistics over all recorded queries.

        Returns
        -------
        dict
            Summary statistics keyed by metric name, each containing
            mean, median, p95, p99, min, max, and count.
        """
        if not self._records:
            return {"count": 0, "metrics": {}}

        metric_names: set[str] = set()
        for record in self._records:
            metric_names.update(k for k in record.keys() if k != "query_id")

        summary: dict[str, Any] = {"count": len(self._records), "metrics": {}}
        for name in sorted(metric_names):
            values = []
            for record in self._records:
                value = record.get(name)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if not values:
                continue

            arr = np.asarray(values, dtype=float)
            summary["metrics"][name] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return summary

    @property
    def records(self) -> list[dict]:
        """Expose collected records for downstream serialization."""
        return list(self._records)
