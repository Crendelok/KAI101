"""pytest-benchmark: timings for the hot paths, plus a machine-independent regression guard.

uv run pytest tests/test_benchmarks.py --benchmark-only
"""

from __future__ import annotations

import random
import time

import pytest

from findex.corpus import Document
from findex.fast import search_fast
from findex.index import Index, build_index
from findex.search import _matching_ids, search

pytestmark = pytest.mark.slow

rng = random.Random(11)
VOCAB = [f"w{i}" for i in range(5000)]
WEIGHTS = [1 / (i + 1) for i in range(len(VOCAB))]
DOCS = [
    Document(i, f"d{i}", f"d{i}", " ".join(rng.choices(VOCAB, WEIGHTS, k=300))) for i in range(3000)
]


@pytest.fixture(scope="module")
def big() -> Index:
    return build_index(DOCS, store_text=False)


def test_bench_build_index(benchmark) -> None:  # type: ignore[no-untyped-def]
    benchmark.pedantic(build_index, args=(DOCS[:500],), kwargs={"store_text": False}, rounds=3)


def test_bench_search_python(benchmark, big: Index) -> None:  # type: ignore[no-untyped-def]
    benchmark(search, big, "w0 OR w1 OR w5")


def test_bench_search_numpy(benchmark, big: Index) -> None:  # type: ignore[no-untyped-def]
    benchmark(search_fast, big, "w0 OR w1 OR w5")


def best_of(fn, *args: object, n: int = 7) -> float:  # type: ignore[no-untyped-def]
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t)
    return min(times)


def test_numpy_scorer_stays_much_faster(big: Index) -> None:
    """Regression guard that holds on any machine: a ratio, not an absolute time."""
    q = "w0 OR w1 OR w5"  # common terms: ~all 3,000 docs scored
    _matching_ids(big, q, "merge")  # warm the cache so both measure scoring only
    slow, fast = best_of(search, big, q), best_of(search_fast, big, q)
    assert slow / fast > 5, f"numpy scorer only {slow / fast:.1f}× faster"
