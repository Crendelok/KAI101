"""Wall time + tracemalloc peak for a block: ``with measured("build") as m: ...``."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


@dataclass
class Measurement:
    label: str
    elapsed: float = 0.0
    peak_bytes: int = 0

    def __str__(self) -> str:
        return f"{self.elapsed:.3f} s, peak {human_bytes(self.peak_bytes)}"


@contextmanager
def measured(label: str, *, trace_memory: bool = True) -> Generator[Measurement]:
    m = Measurement(label)
    was_tracing = tracemalloc.is_tracing()
    if trace_memory and not was_tracing:
        tracemalloc.start()
    if trace_memory:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield m
    finally:
        m.elapsed = time.perf_counter() - t0
        if trace_memory:
            m.peak_bytes = tracemalloc.get_traced_memory()[1]
            if not was_tracing:
                tracemalloc.stop()
