from __future__ import annotations

import os
import time
import functools
from typing import Callable, Any, TypeVar, cast
from loguru import logger
from ..config import settings


F = TypeVar("F", bound=Callable[..., Any])


def setup_logging(level: str | None = None) -> None:
    os.makedirs(settings.log_dir, exist_ok=True)
    log_file = os.path.join(settings.log_dir, "ra_crew.log")
    logger.remove()
    lvl = level or settings.log_level
    logger.add(lambda msg: print(msg, end=""), level=lvl)
    logger.add(log_file, rotation="5 MB", retention="10 days", level=lvl)


def timeit(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"{func.__name__} took {elapsed:.1f} ms")

    return cast(F, wrapper)
