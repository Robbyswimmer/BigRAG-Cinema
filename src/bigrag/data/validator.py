"""
bigrag.data.validator -- Schema validation and deduplication.

Ensures that incoming DataFrames conform to the expected schema
(column names, types, non-null constraints) and removes duplicate rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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
    raise NotImplementedError("validate_dataframe is not yet implemented")
