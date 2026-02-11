"""
bigrag.benchmark.experiment_runner -- End-to-end experiment orchestration.

Loads data, warms up the cluster, runs the workload against each
strategy / data fraction, collects metrics, and saves results.
"""

from __future__ import annotations


def run_experiment(config: dict) -> dict:
    """Execute a full benchmark experiment defined by *config*.

    Parameters
    ----------
    config : dict
        Experiment configuration containing keys such as
        ``"data_path"``, ``"strategies"``, ``"fractions"``,
        ``"num_queries"``, ``"warmup_rounds"``, and ``"output_dir"``.

    Returns
    -------
    dict
        Nested results structure keyed by strategy name and fraction,
        containing raw and summary metrics.
    """
    raise NotImplementedError("run_experiment is not yet implemented")
