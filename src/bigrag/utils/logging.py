"""
bigrag.utils.logging -- Python logging setup.

Configures the root (or package-level) logger with handlers, formatters,
and log levels defined in an external YAML/dict configuration.
"""

from __future__ import annotations

from pathlib import Path


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
    raise NotImplementedError("setup_logging is not yet implemented")
