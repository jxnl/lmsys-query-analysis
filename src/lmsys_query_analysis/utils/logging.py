"""Logging utilities with Rich handler and simple setup."""
from __future__ import annotations

import logging
from typing import Optional
from rich.logging import RichHandler


def setup_logging(verbose: bool = False, level: Optional[int] = None) -> None:
    """Configure global logging with Rich handler.

    Args:
        verbose: When True, set level to DEBUG; otherwise INFO.
        level: Optional explicit level to override verbose flag.
    """
    resolved_level = level if level is not None else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    # Reduce noise from third-party libs by default
    for noisy in ("sqlalchemy.engine", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

