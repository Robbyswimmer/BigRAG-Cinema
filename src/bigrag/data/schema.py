"""
bigrag.data.schema -- Shared column names, types, and embedding constants.

Single source of truth for the canonical schema used across data
ingestion, Spark processing, and benchmark analysis.
"""

# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# Column name constants  (Amazon Reviews 2023 schema)
# ---------------------------------------------------------------------------
COL_RATING: str = "rating"
COL_TITLE: str = "title"
COL_TEXT: str = "text"
COL_ASIN: str = "asin"
COL_PARENT_ASIN: str = "parent_asin"
COL_USER_ID: str = "user_id"
COL_TIMESTAMP: str = "timestamp"
COL_HELPFUL_VOTE: str = "helpful_vote"
COL_VERIFIED_PURCHASE: str = "verified_purchase"
COL_EMBEDDING: str = "embedding"

# Ordered list of all expected columns (before embedding is attached).
EXPECTED_COLUMNS: list = [
    COL_RATING,
    COL_TITLE,
    COL_TEXT,
    COL_ASIN,
    COL_PARENT_ASIN,
    COL_USER_ID,
    COL_TIMESTAMP,
    COL_HELPFUL_VOTE,
    COL_VERIFIED_PURCHASE,
]
