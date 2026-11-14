"""The race a shared-state threaded indexer would have: ``counter[term] += 1``.

    python scripts/race_demo.py            # GIL build
    python3.13t scripts/race_demo.py       # free-threaded build

findex's real threaded build has no shared mutable state (each chunk builds its
own PartialIndex), so it is correct on both builds. This script shows what goes
wrong if you *do* share one Counter between threads — and the two fixes.
"""

from __future__ import annotations

import dis
import sys
import threading
from collections import Counter

THREADS, N, KEYS = 8, 200_000, 4


def unsafe(counter: Counter[int]) -> None:
    for i in range(N):
        counter[i % KEYS] += 1  # load, add, store: a thread switch in between loses an update


def locked(counter: Counter[int], lock: threading.Lock) -> None:
    for i in range(N):
        with lock:
            counter[i % KEYS] += 1


def private() -> Counter[int]:
    mine: Counter[int] = Counter()  # no sharing at all: merge at the end
    for i in range(N):
        mine[i % KEYS] += 1
    return mine


def run(target, *args) -> None:  # type: ignore[no-untyped-def]
    ts = [threading.Thread(target=target, args=args) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main() -> None:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, GIL {'enabled' if gil else 'DISABLED'}")
    expected = THREADS * N
    print("bytecode of `counter[k] += 1`:")
    dis.dis(compile("counter[k] += 1", "<demo>", "exec"))

    c: Counter[int] = Counter()
    run(unsafe, c)
    lost = expected - c.total()
    print(f"shared Counter, no lock : {c.total():>9,} of {expected:,}  (lost {lost:,})")

    c = Counter()
    run(locked, c, threading.Lock())
    print(f"shared Counter + Lock   : {c.total():>9,} of {expected:,}")

    parts: list[Counter[int]] = []
    ts = [threading.Thread(target=lambda: parts.append(private())) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    merged = sum(parts, Counter())
    print(f"private Counters, merged: {merged.total():>9,} of {expected:,}")


if __name__ == "__main__":
    main()
