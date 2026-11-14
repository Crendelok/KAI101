"""Lab 2 measurements: posting representations, merge vs set, storage formats.

python scripts/bench_lab02.py data/sections
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import time
import timeit
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.measure import human_bytes, measured
from findex.setops import intersect, union
from findex.store import load, save


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlotPosting:
    doc_id: int
    tf: int


def build_objects(root: Path, cls: type) -> dict[str, list[object]]:
    postings: defaultdict[str, list[object]] = defaultdict(list)
    for doc in iter_documents(root):
        from findex.tokenize import tokenize

        for term, tf in Counter(tokenize(doc.text)).items():
            postings[term].append(cls(doc.doc_id, tf))
    return dict(postings)


def build_arrays(root: Path) -> dict[str, tuple[array[int], array[int]]]:
    from findex.tokenize import tokenize

    postings: dict[str, tuple[array[int], array[int]]] = {}
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            pair = postings.get(term)
            if pair is None:
                pair = postings[term] = (array("I"), array("I"))
            pair[0].append(doc.doc_id)
            pair[1].append(tf)
    return postings


def memory_study(root: Path, tmp: Path) -> None:
    print("\n### Postings representation\n")
    print("| Postings representation | Peak memory (build) | Index file size | Load time |")
    print("|---|---|---|---|")
    variants = [
        ("`list[Posting]` with plain `@dataclass`", lambda: build_objects(root, PlainPosting)),
        ("`list[Posting]` with `slots=True`", lambda: build_objects(root, SlotPosting)),
        ("`array('I')` pairs (doc_ids, tfs)", lambda: build_arrays(root)),
    ]
    for label, build in variants:
        with measured(label) as m:
            postings = build()
        f = tmp / "variant.pkl"
        with f.open("wb") as fh:
            pickle.dump(postings, fh, protocol=pickle.HIGHEST_PROTOCOL)
        del postings
        t0 = time.perf_counter()
        with f.open("rb") as fh:
            pickle.load(fh)  # noqa: S301 — our own file
        load_s = time.perf_counter() - t0
        print(
            f"| {label} | {human_bytes(m.peak_bytes)} | "
            f"{human_bytes(f.stat().st_size)} | {load_s:.2f} s |"
        )


def engines(root: Path) -> None:
    index = build_index(iter_documents(root), store_text=False)
    by_df = sorted(index, key=index.df)
    common = by_df[-2:]
    rare = [t for t in by_df if index.df(t) >= 2][:2]
    mid = [t for t in by_df if 50 <= index.df(t) <= 60][:1]
    print(f"\n### merge vs set ({index.num_docs:,} docs)\n")
    print("| Query | df | Op | merge (µs) | set (µs) | Faster |")
    print("|---|---|---|---|---|---|")
    cases = [
        (common, "AND"),
        (common, "OR"),
        (rare, "AND"),
        (rare, "OR"),
        ([common[-1], mid[0]], "AND"),
    ]
    for (a, b), op in cases:
        fn = intersect if op == "AND" else union
        x, y = index[a].doc_ids, index[b].doc_ids
        times = {}
        for engine in ("merge", "set"):
            n, total = timeit.Timer(lambda e=engine, f=fn, x=x, y=y: f(x, y, e)).autorange()
            times[engine] = total / n * 1e6
        best = min(times, key=times.__getitem__)
        print(
            f"| `{a}` {op} `{b}` | {len(x):,} / {len(y):,} | {op} | {times['merge']:,.0f} | "
            f"{times['set']:,.0f} | {best} ({max(times.values()) / min(times.values()):.1f}×) |"
        )


def formats(root: Path, tmp: Path) -> None:
    print("\n### Storage formats\n")
    print("| Index | Format | File size | Save | Load |")
    print("|---|---|---|---|---|")
    for positions in (False, True):
        with measured("build") as m:
            index = build_index(iter_documents(root), positions=positions)
        label = "with positions" if positions else "no positions"
        print(f"| {label} (build peak {human_bytes(m.peak_bytes)}) | | | | |")
        for name, fmt, lazy in (
            ("pickle", "x.pkl", False),
            (".fidx eager", "x.fidx", False),
            (".fidx mmap", "x.fidx", True),
        ):
            path = tmp / fmt
            t0 = time.perf_counter()
            save(index, path)
            save_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            loaded = load(path, lazy=lazy)
            load_s = time.perf_counter() - t0
            loaded.close()
            print(
                f"| | {name} | {human_bytes(path.stat().st_size)} | {save_s:.2f} s | "
                f"{load_s * 1000:,.0f} ms |"
            )


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sections")
    with tempfile.TemporaryDirectory() as d:
        memory_study(root, Path(d))
        engines(root)
        formats(root, Path(d))


if __name__ == "__main__":
    main()
