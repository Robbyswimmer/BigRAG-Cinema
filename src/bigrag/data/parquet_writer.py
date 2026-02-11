"""
bigrag.data.parquet_writer -- Join embeddings and write Parquet files.

Merges the embedding vectors back into the source DataFrame, then
writes out the full Parquet file as well as fractional subsets (e.g.
25%, 50%, 75%) for scaling experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def write_parquet_fractions(
    df: "pd.DataFrame",
    embeddings: "np.ndarray",
    output_dir: Path,
    fractions: Sequence[float] = (0.25, 0.50, 0.75, 1.0),
) -> list[Path]:
    """Join *embeddings* onto *df* and write Parquet files for each fraction.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame (validated, deduplicated).
    embeddings : np.ndarray
        Embedding matrix aligned row-wise with *df*.
    output_dir : Path
        Directory where Parquet files will be written.
    fractions : Sequence[float]
        Dataset fractions to materialise (e.g. 0.25 for 25%).

    Returns
    -------
    list[Path]
        Paths to the written Parquet files.
    """
    raise NotImplementedError("write_parquet_fractions is not yet implemented")
