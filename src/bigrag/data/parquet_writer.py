"""
bigrag.data.parquet_writer -- Join embeddings and write Parquet files.

Merges the embedding vectors back into the source DataFrame, then
writes out the full Parquet file as well as fractional subsets (e.g.
25%, 50%, 75%) for scaling experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bigrag.data.schema import COL_EMBEDDING
from bigrag.data.validator import validate_dataframe


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
    if len(df) != len(embeddings):
        raise ValueError(
            "Embeddings row count must match DataFrame row count "
            f"({len(embeddings)} != {len(df)})."
        )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with_embeddings = df.copy()
    with_embeddings[COL_EMBEDDING] = embeddings.astype(np.float32).tolist()

    paths: list[Path] = []
    for frac in sorted(set(float(f) for f in fractions)):
        if frac <= 0 or frac > 1.0:
            raise ValueError(f"Fraction must be in (0, 1], got {frac}.")
        subset_rows = max(1, int(len(with_embeddings) * frac))
        subset = with_embeddings.iloc[:subset_rows].reset_index(drop=True)
        filename = f"reviews_{int(round(frac * 100)):03d}.parquet"
        path = output / filename
        subset.to_parquet(path, index=False)
        paths.append(path)
    return paths


def write_parquet(
    csv_path: str | Path,
    embeddings_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Backward-compatible wrapper for the CLI script."""
    source_csv = Path(csv_path).expanduser().resolve()
    source_embeddings = Path(embeddings_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()

    if not source_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {source_csv}")
    if not source_embeddings.exists():
        raise FileNotFoundError(f"Embeddings file does not exist: {source_embeddings}")

    raw_df = pd.read_csv(source_csv)
    cleaned_df, _ = validate_dataframe(raw_df)
    embeddings = np.load(source_embeddings)

    if len(cleaned_df) != len(embeddings):
        raise ValueError(
            "Validated DataFrame length must match embeddings length "
            f"({len(cleaned_df)} != {len(embeddings)})."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    enriched = cleaned_df.copy()
    enriched[COL_EMBEDDING] = embeddings.astype(np.float32).tolist()
    enriched.to_parquet(target, index=False)
    return target
