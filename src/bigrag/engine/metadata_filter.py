"""
bigrag.engine.metadata_filter -- Metadata filter builders for Spark queries.

Constructs Spark Column filter expressions from user-supplied
criteria such as time ranges, review scores, and user-ID lists.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import Column


def build_filters(
    time_range: Optional[Tuple[str, str]] = None,
    score_range: Optional[Tuple[float, float]] = None,
    user_ids: Optional[Sequence[str]] = None,
) -> "Column":
    """Compose a conjunctive Spark filter from the supplied criteria.

    Parameters
    ----------
    time_range : tuple[str, str] | None
        ``(start_iso, end_iso)`` bounding the ``timestamp`` column.
    score_range : tuple[float, float] | None
        ``(min_score, max_score)`` bounding the ``score`` column.
    user_ids : Sequence[str] | None
        Allowlist of user IDs to include.

    Returns
    -------
    pyspark.sql.Column
        A combined boolean Column expression (all conditions ANDed).
    """
    raise NotImplementedError("build_filters is not yet implemented")
