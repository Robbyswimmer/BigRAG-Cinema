"""
bigrag.strategies -- Pluggable execution strategies for hybrid search.

Each strategy implements a different ordering of vector search and
metadata filtering, enabling benchmark comparisons.  Strategies are
registered in ``strategies.registry`` and resolved by name at runtime.
"""
