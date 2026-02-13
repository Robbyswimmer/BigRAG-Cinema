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

import numpy as np

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
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = Path(base).expanduser() / candidate
    return candidate.resolve()


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
    import pandas as pd

    parquet_path = Path(path).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    return pd.read_parquet(parquet_path)


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
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class _Encoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            return super().default(o)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=_Encoder, ensure_ascii=False)
