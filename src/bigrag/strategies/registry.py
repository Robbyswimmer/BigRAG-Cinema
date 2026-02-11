"""
bigrag.strategies.registry -- Name-to-strategy class mapping.

Maintains a registry dict that maps human-readable strategy names
(e.g. ``"filter_first"``) to their corresponding class objects,
and exposes a convenience lookup function.
"""

from __future__ import annotations

from typing import Dict, Type

from bigrag.strategies.base import ExecutionStrategy
from bigrag.strategies.filter_first import FilterFirstStrategy
from bigrag.strategies.vector_first import VectorFirstStrategy
from bigrag.strategies.hybrid_parallel import HybridParallelStrategy
from bigrag.strategies.adaptive import AdaptiveStrategy

STRATEGY_REGISTRY: Dict[str, Type[ExecutionStrategy]] = {
    "filter_first": FilterFirstStrategy,
    "vector_first": VectorFirstStrategy,
    "hybrid_parallel": HybridParallelStrategy,
    "adaptive": AdaptiveStrategy,
}


def get_strategy(name: str) -> ExecutionStrategy:
    """Look up *name* in the registry and return an instance.

    Parameters
    ----------
    name : str
        Registered strategy name (e.g. ``"filter_first"``).

    Returns
    -------
    ExecutionStrategy
        An instance of the corresponding strategy class.

    Raises
    ------
    KeyError
        If *name* is not found in ``STRATEGY_REGISTRY``.
    """
    raise NotImplementedError("get_strategy is not yet implemented")
