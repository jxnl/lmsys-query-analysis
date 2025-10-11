"""Logging utilities with Rich handler and simple setup."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from rich.logging import RichHandler

F = TypeVar('F', bound=Callable[..., Any])


def setup_logging(verbose: bool = False, level: Optional[int] = None) -> None:
    """Configure global logging with Rich handler.

    Args:
        verbose: When True, set level to DEBUG; otherwise INFO.
        level: Optional explicit level to override verbose flag.
    """
    resolved_level = (
        level if level is not None else (logging.DEBUG if verbose else logging.INFO)
    )
    logging.basicConfig(
        level=resolved_level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    # Reduce noise from third-party libs by default
    for noisy in ("sqlalchemy.engine", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def trace(func: F) -> F:
    """Decorator that logs function execution time.
    
    Args:
        func: The function to wrap with timing logs.
        
    Returns:
        The wrapped function with timing logs.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"[dim]⏱️  `{func.__name__}` completed in {elapsed:.3f}s[/dim]")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[dim]⏱️  `{func.__name__}` failed after {elapsed:.3f}s: {e}[/dim]")
            raise
    
    return wrapper  # type: ignore
