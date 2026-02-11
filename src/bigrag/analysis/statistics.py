"""
bigrag.analysis.statistics -- Percentiles, confidence intervals, significance tests.

Provides statistical utility functions used to summarise benchmark
results and determine whether observed differences between strategies
are statistically significant.
"""

from __future__ import annotations

from typing import Sequence


def compute_percentiles(
    values: Sequence[float],
    percentiles: Sequence[float] = (50, 90, 95, 99),
) -> dict[float, float]:
    """Compute the requested percentiles for *values*.

    Parameters
    ----------
    values : Sequence[float]
        Raw measurement values.
    percentiles : Sequence[float]
        Percentile ranks to compute (0--100 scale).

    Returns
    -------
    dict[float, float]
        Mapping from percentile rank to computed value.
    """
    raise NotImplementedError("compute_percentiles is not yet implemented")


def confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a confidence interval for the mean of *values*.

    Parameters
    ----------
    values : Sequence[float]
        Sample measurements.
    confidence : float
        Confidence level (e.g. 0.95 for 95%).

    Returns
    -------
    tuple[float, float]
        ``(lower_bound, upper_bound)`` of the CI.
    """
    raise NotImplementedError("confidence_interval is not yet implemented")


def significance_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """Run a two-sample significance test (e.g. Welch's t-test).

    Parameters
    ----------
    sample_a : Sequence[float]
        Measurements from condition A.
    sample_b : Sequence[float]
        Measurements from condition B.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict
        Keys: ``"statistic"``, ``"p_value"``, ``"significant"`` (bool).
    """
    raise NotImplementedError("significance_test is not yet implemented")
