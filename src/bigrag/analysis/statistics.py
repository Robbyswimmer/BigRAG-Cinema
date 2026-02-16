"""
bigrag.analysis.statistics -- Percentiles, confidence intervals, significance tests.

Provides statistical utility functions used to summarise benchmark
results and determine whether observed differences between strategies
are statistically significant.
"""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import NormalDist
from typing import Sequence

import numpy as np


def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must contain only finite numeric values")
    return arr


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
    arr = _as_array(values)
    result: dict[float, float] = {}
    for p in percentiles:
        p_float = float(p)
        if p_float < 0 or p_float > 100:
            raise ValueError(f"Percentile must be between 0 and 100: {p}")
        result[p_float] = float(np.percentile(arr, p_float))
    return result


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
    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be between 0 and 1")

    arr = _as_array(values)
    n = arr.size
    mean = float(np.mean(arr))
    if n == 1:
        return (mean, mean)

    std = float(np.std(arr, ddof=1))
    if std == 0:
        return (mean, mean)

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = z * std / sqrt(float(n))
    return (mean - margin, mean + margin)


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
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")

    a = _as_array(sample_a)
    b = _as_array(sample_b)
    if a.size < 2 or b.size < 2:
        raise ValueError("Both samples must have at least 2 values")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    n_a = float(a.size)
    n_b = float(b.size)

    denom = sqrt((var_a / n_a) + (var_b / n_b))
    if denom == 0 or not isfinite(denom):
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (mean_a - mean_b) / denom
        # Approximate two-sided p-value using normal CDF.
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(statistic)))

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "n_a": int(a.size),
        "n_b": int(b.size),
    }
