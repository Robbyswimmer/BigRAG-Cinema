"""
bigrag.utils.config -- YAML configuration loader.

Reads a YAML file and returns its contents as a plain Python dict,
with optional schema validation and environment-variable interpolation.
"""

from __future__ import annotations

from pathlib import Path


def load_config(path: Path) -> dict:
    """Load and return the YAML configuration at *path*.

    Parameters
    ----------
    path : Path
        Path to a ``.yaml`` / ``.yml`` configuration file.

    Returns
    -------
    dict
        Parsed configuration as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    raise NotImplementedError("load_config is not yet implemented")
