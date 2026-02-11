"""
bigrag.analysis.plotting -- Latency CDFs, throughput bars, scaling curves.

Generates matplotlib / seaborn figures suitable for inclusion in the
project report and presentation slides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def plot_latency_cdf(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    """Plot cumulative distribution functions of per-query latency.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics keyed by strategy name.
    output_path : Path | None
        If provided, save the figure to this path.
    """
    raise NotImplementedError("plot_latency_cdf is not yet implemented")


def plot_throughput_bars(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    """Plot a grouped bar chart of throughput (queries/sec) per strategy.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics keyed by strategy name.
    output_path : Path | None
        If provided, save the figure to this path.
    """
    raise NotImplementedError("plot_throughput_bars is not yet implemented")


def plot_scaling_curves(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    """Plot latency vs. data fraction scaling curves per strategy.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics keyed by (strategy, fraction).
    output_path : Path | None
        If provided, save the figure to this path.
    """
    raise NotImplementedError("plot_scaling_curves is not yet implemented")
