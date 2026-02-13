"""
bigrag.engine.metadata_filter -- Metadata filter builders for Spark queries.

Constructs Spark Column filter expressions from user-supplied
criteria such as time ranges, review scores, and user-ID lists.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

from pyspark.sql import functions as F

from bigrag.data.schema import COL_RATING, COL_TIMESTAMP, COL_USER_ID

if TYPE_CHECKING:
    from pyspark.sql import Column


def _to_unix_seconds(value: str) -> int:
    """Convert a string timestamp (epoch or ISO-8601) to unix seconds."""
    raw = str(value).strip()
    if raw.isdigit():
        return int(raw)
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


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
    conditions: list["Column"] = []

    if time_range is not None:
        start_raw, end_raw = time_range
        start_ts = _to_unix_seconds(start_raw)
        end_ts = _to_unix_seconds(end_raw)
        if start_ts > end_ts:
            raise ValueError("time_range start must be <= end")
        conditions.append(F.col(COL_TIMESTAMP).cast("long").between(start_ts, end_ts))

    if score_range is not None:
        min_score, max_score = score_range
        if min_score > max_score:
            raise ValueError("score_range min must be <= max")
        conditions.append(
            F.col(COL_RATING).cast("double").between(float(min_score), float(max_score))
        )

    if user_ids:
        user_list = [str(u) for u in user_ids if str(u).strip()]
        if user_list:
            conditions.append(F.col(COL_USER_ID).isin(user_list))

    if not conditions:
        return F.lit(True)

    combined = conditions[0]
    for condition in conditions[1:]:
        combined = combined & condition
    return combined
