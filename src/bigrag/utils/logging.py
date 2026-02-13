"""
bigrag.utils.logging -- Python logging setup.

Configures the root (or package-level) logger with handlers, formatters,
and log levels defined in an external YAML/dict configuration.
"""

from __future__ import annotations

import logging.config
from pathlib import Path

import yaml


def setup_logging(config_path: Path) -> None:
    """Configure Python logging from the YAML file at *config_path*.

    Parameters
    ----------
    config_path : Path
        Path to a YAML file whose structure matches
        ``logging.config.dictConfig`` expectations.

    Returns
    -------
    None
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Logging config does not exist: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    handlers = config.get("handlers", {})
    for handler in handlers.values():
        filename = handler.get("filename")
        if filename:
            Path(filename).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
