"""The five "Prove it to yourself" REPL experiments from Lab 01, runnable in one go.

    python scripts/repl_experiments.py
"""

import tempfile
import tracemalloc
import unicodedata
from pathlib import Path


def banner(n: int, title: str) -> None:
    print(f"\n--- {n}. {title} ---")


banner(1, "a list vs. an iterator over it")
xs = [1, 2, 3]
it = iter(xs)
print(next(it), xs)  # 1 [1, 2, 3]
print(list(it))      # [2, 3]  — the finger continues from where it was
print(list(it))      # []      — same finger, already past the end
print(xs)            # [1, 2, 3] — the box was never touched

banner(2, "peak memory: list comprehension vs. generator expression")
for label, make in (
    ("list", lambda: sum([x * x for x in range(10**7)])),  # noqa: C419
    ("gen ", lambda: sum(x * x for x in range(10**7))),
):
    tracemalloc.start()
    make()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label}: peak {peak / 2**20:8.2f} MiB")
# Lazy: only the generator frame, the current int, and the running sum. range stores 3 ints.

banner(3, "a call that runs vs. a call that builds a generator")


def ordinary(n):
    print("start", n)
    return n


def countdown(n):
    print("start", n)
    while n > 0:
        yield n
        n -= 1


ordinary(3)               # prints "start 3" now
g = countdown(3)          # prints nothing: the body hasn't started
print(type(g))
print(next(g))            # now "start 3", then 3

banner(4, "NFC normalization")
a, b = "café", "café"
print(a, b, a == b)                                   # café café False
nfc = unicodedata.normalize
print(nfc("NFC", a) == nfc("NFC", b), len(a), len(b))  # True 4 5

banner(5, "a file is a single-pass iterator")
with tempfile.TemporaryDirectory() as d:
    p = Path(d, "f.txt")
    p.write_text("one\ntwo\n", encoding="utf-8")
    with p.open(encoding="utf-8") as f:
        print("first pass :", [line.rstrip() for line in f])
        print("second pass:", [line.rstrip() for line in f])  # [] — cursor at EOF (Pitfall 1)
        f.seek(0)
        print("after seek :", [line.rstrip() for line in f])
