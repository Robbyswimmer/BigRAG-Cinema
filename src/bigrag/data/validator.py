"""
bigrag.data.validator -- Schema validation and deduplication.

Ensures that incoming DataFrames conform to the expected schema
(column names, types, non-null constraints) and removes duplicate rows.
"""

from __future__ import annotations

import pandas as pd

from bigrag.data.schema import (
    COL_HELPFUL_VOTE,
    COL_RATING,
    COL_TEXT,
    COL_TIMESTAMP,
    COL_USER_ID,
    COL_VERIFIED_PURCHASE,
    EXPECTED_COLUMNS,
)

def validate_dataframe(df: "pd.DataFrame") -> tuple:
    """Validate *df* against the canonical schema and deduplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw DataFrame loaded from CSV.

    Returns
    -------
    tuple
        A two-element tuple ``(cleaned_df, report)`` where *cleaned_df*
        is the validated / deduplicated DataFrame and *report* is a dict
        of validation statistics (rows dropped, issues found, etc.).
    """
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    working = df.copy()
    initial_rows = len(working)

    working[COL_RATING] = pd.to_numeric(working[COL_RATING], errors="coerce")
    working[COL_TIMESTAMP] = pd.to_numeric(working[COL_TIMESTAMP], errors="coerce")
    working[COL_HELPFUL_VOTE] = pd.to_numeric(working[COL_HELPFUL_VOTE], errors="coerce")

    bool_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    working[COL_VERIFIED_PURCHASE] = (
        working[COL_VERIFIED_PURCHASE]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(bool_map)
    )

    working[COL_TEXT] = working[COL_TEXT].astype(str).str.strip()
    working[COL_USER_ID] = working[COL_USER_ID].astype(str).str.strip()

    required_non_null = [COL_TEXT, COL_USER_ID, COL_TIMESTAMP, COL_RATING]
    invalid_before = len(working)
    working = working.dropna(subset=required_non_null)
    working = working[working[COL_TEXT] != ""]
    invalid_rows = invalid_before - len(working)

    before_dedup = len(working)
    working = working.drop_duplicates(subset=EXPECTED_COLUMNS)
    duplicates_removed = before_dedup - len(working)

    cleaned = working[EXPECTED_COLUMNS].reset_index(drop=True)
    report = {
        "initial_rows": initial_rows,
        "invalid_rows": invalid_rows,
        "duplicates_removed": duplicates_removed,
        "final_rows": len(cleaned),
    }
    return cleaned, report
