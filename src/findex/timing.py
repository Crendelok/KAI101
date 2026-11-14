"""The ``@timed`` decorator (Lab 3): logs how long a call took, in ms."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger("findex.timing")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Log ``<qualname> took N.N ms`` at DEBUG after every call."""

    @functools.wraps(fn)  # keep __name__, __doc__, __wrapped__, the signature
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.1f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)

    return wrapper
