"""
bigrag.utils.config -- YAML configuration loader.

Reads a YAML file and returns its contents as a plain Python dict,
with optional schema validation and environment-variable interpolation.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


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
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    rendered = os.path.expandvars(config_path.read_text(encoding="utf-8"))
    parsed = yaml.safe_load(rendered) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration file must parse to a dictionary: {config_path}")
    return parsed
