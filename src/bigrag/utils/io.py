"""
bigrag.utils.io -- Path resolution, Parquet helpers, and JSON serialization.

Convenience wrappers for common file-system operations used across
the project, including safe path construction, Parquet reading, and
JSON output with custom encoders for numpy types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def resolve_path(raw: str, base: Path | None = None) -> Path:
    """Resolve *raw* to an absolute ``Path``, optionally relative to *base*.

    Parameters
    ----------
    raw : str
        Raw path string (may be relative or contain ``~``).
    base : Path | None
        If provided, relative paths are resolved against *base*.

    Returns
    -------
    Path
        Fully resolved absolute path.
    """
    raise NotImplementedError("resolve_path is not yet implemented")


def read_parquet(path: Path) -> "pd.DataFrame":
    """Read a Parquet file into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        Path to the ``.parquet`` file.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame.
    """
    raise NotImplementedError("read_parquet is not yet implemented")


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Serialize *data* to JSON and write it to *path*.

    Handles numpy scalar types and ``Path`` objects via a custom encoder.

    Parameters
    ----------
    data : Any
        JSON-serialisable data (dict, list, etc.).
    path : Path
        Destination file path.
    indent : int
        JSON indentation level.
    """
    raise NotImplementedError("save_json is not yet implemented")
