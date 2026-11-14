"""M3 + M4 — the stats pipeline, and its deliberately eager twin.

python -m findex.stats data/                  # lazy pipeline, top-50 terms
python -m findex.stats data/ --limit 100      # first 100 docs (itertools.islice)
python -m findex.stats data/ --eager          # the list-based version
python -m findex.stats data/ --compare        # both, as a README table
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class Stats:
    documents: int
    tokens: int
    counts: Counter[str]

    @property
    def vocabulary(self) -> int:
        return len(self.counts)


def lazy_stats(root: Path, limit: int | None = None) -> Stats:
    """One pass, constant memory: only the current document + the Counter live."""
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    n_docs = 0
    counts: Counter[str] = Counter()
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))  # Counter pulls tokens one at a time
    return Stats(n_docs, counts.total(), counts)


def eager_stats(root: Path, limit: int | None = None) -> Stats:
    """The anti-pattern, on purpose: materialize every stage as a list."""
    docs = list(iter_documents(root))
    if limit is not None:
        docs = docs[:limit]
    texts = [doc.text for doc in docs]
    token_lists = [list(tokenize(text)) for text in texts]
    all_tokens = [tok for toks in token_lists for tok in toks]
    counts = Counter(all_tokens)
    return Stats(len(docs), len(all_tokens), counts)


@dataclass
class Measured:
    stats: Stats
    elapsed: float
    peak_bytes: int


def measure(fn: Callable[[Path, int | None], Stats], root: Path, limit: int | None) -> Measured:
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        stats = fn(root, limit)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Measured(stats, elapsed, peak)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def print_report(label: str, m: Measured, top: int) -> None:
    s = m.stats
    print(f"== {label} ==")
    print(f"documents       {s.documents:>14,}")
    print(f"total tokens    {s.tokens:>14,}")
    print(f"vocabulary      {s.vocabulary:>14,}")
    print(f"elapsed         {m.elapsed:>12.2f} s")
    print(f"peak memory     {human_bytes(m.peak_bytes):>14}")
    if top:
        print(f"\ntop {top} terms:")
        width = max((len(t) for t, _ in s.counts.most_common(top)), default=0)
        for rank, (term, n) in enumerate(s.counts.most_common(top), start=1):
            print(f"{rank:>4}. {term:<{width}}  {n:>12,}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m findex.stats",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", type=Path, help="corpus dir or file (.txt/.jsonl, opt. .gz/.bz2)")
    p.add_argument("--limit", type=int, default=None, help="only the first N documents")
    p.add_argument("--top", type=int, default=50, help="how many top terms to print (default 50)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eager", action="store_true", help="run the list-based version")
    mode.add_argument("--compare", action="store_true", help="run both, print a markdown table")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Windows consoles default to a legacy code page; the top terms may be Cyrillic.
    stdout: object = sys.stdout
    if isinstance(stdout, io.TextIOWrapper):
        # The buffer type parameter is unknowable here; reconfigure doesn't depend on it.
        stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportUnknownMemberType]

    if args.compare:
        eager = measure(eager_stats, args.root, args.limit)
        lazy = measure(lazy_stats, args.root, args.limit)
        assert eager.stats.counts == lazy.stats.counts, "eager and lazy disagree!"
        print("| Version | Documents | Tokens | Peak memory | Elapsed |")
        print("|---|---|---|---|---|")
        for label, m in (("eager (lists)", eager), ("lazy (generators)", lazy)):
            print(
                f"| {label} | {m.stats.documents:,} | {m.stats.tokens:,} "
                f"| {human_bytes(m.peak_bytes)} | {m.elapsed:.2f} s |"
            )
        return 0

    fn, label = (eager_stats, "eager") if args.eager else (lazy_stats, "lazy")
    print_report(label, measure(fn, args.root, args.limit), args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
