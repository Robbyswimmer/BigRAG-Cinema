"""
bigrag.benchmark.timer -- High-resolution timing context manager.

Provides a lightweight ``Timer`` class that uses ``time.perf_counter``
to measure wall-clock elapsed time within a ``with`` block.
"""

from __future__ import annotations

import time


class Timer:
    """Context manager for high-resolution wall-clock timing.

    Usage
    -----
    >>> with Timer() as t:
    ...     do_work()
    >>> print(t.elapsed_s)
    """

    def __init__(self) -> None:
        self.elapsed_s: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_s = time.perf_counter() - self._start
