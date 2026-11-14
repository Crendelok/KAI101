"""Parallel indexing (Lab 5): build partial indexes per chunk of paths, then merge.

The work is split by *paths* (cheap to send to a worker), never by loaded
documents (expensive to pickle). Each chunk is indexed with local doc ids
``0..n-1``; ``merge`` shifts every chunk by the number of documents before it,
so the result is identical to a serial build over the same sorted path list.

``build_partial`` lives at module level because ``ProcessPoolExecutor`` pickles
the *function reference* (module + name) and re-imports it in the worker; a
lambda or nested function can't be pickled. Under ``spawn`` (Windows, macOS)
each worker re-imports ``__main__``, so pool creation must sit behind
``if __name__ == "__main__"`` — the CLI entry point satisfies that.
"""

from __future__ import annotations

import logging
import os
import time
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from findex.corpus import iter_documents_from, iter_paths
from findex.index import DocMeta, Index, PostingList, build_index
from findex.timing import timed

log = logging.getLogger(__name__)

type ExecutorName = Literal["serial", "threads", "processes"]


@dataclass(slots=True)
class PartialIndex:
    """A compact, picklable index over one chunk. Arrays pickle as raw bytes."""

    postings: dict[str, tuple[array[int], array[int], array[int] | None]]
    doc_meta: list[tuple[str, str]]
    doc_lengths: array[int]
    texts: list[bytes] | None
    has_positions: bool
    cpu_seconds: float = 0.0  # CPU time of the thread that built this chunk
    worker_pid: int = 0
    worker_cpu_total: float = 0.0  # cumulative process CPU of the worker at the end of the task


def build_partial(
    paths: Sequence[Path], positions: bool = True, store_text: bool = True
) -> PartialIndex:
    """Index one chunk of files. Module-level and argument-only: safe for any executor."""
    cpu0 = time.thread_time()
    ix = build_index(iter_documents_from(paths), positions=positions, store_text=store_text)
    postings = {t: (pl.doc_ids, pl.tfs, pl.positions) for t, pl in ix.items()}
    blobs = ix._texts  # pyright: ignore[reportPrivateUsage]  # same package, internal hand-off
    texts = [blobs[i] for i in range(ix.num_docs)] if blobs is not None else None
    return PartialIndex(
        postings,
        [(m.path, m.title) for m in ix.doc_meta],
        array("I", ix.doc_lengths),
        texts,
        positions,
        time.thread_time() - cpu0,
        os.getpid(),
        time.process_time(),
    )


@timed
def merge(partials: Iterable[PartialIndex]) -> Index:
    """Concatenate partials in order, shifting each chunk's doc ids by an offset."""
    postings: dict[str, PostingList] = {}
    doc_meta: list[DocMeta] = []
    doc_lengths = array("I")
    texts: list[bytes] | None = []
    has_positions = True
    for part in partials:
        offset = len(doc_meta)
        has_positions = part.has_positions
        for term, (ids, tfs, pos) in part.postings.items():
            pl = postings.get(term)
            if pl is None:
                pl = postings[term] = PostingList(
                    array("I"), array("I"), array("I") if pos is not None else None
                )
            if offset:
                pl.doc_ids.extend(array("I", [d + offset for d in ids]))
            else:
                pl.doc_ids.extend(ids)
            pl.tfs.extend(tfs)
            if pos is not None and pl.positions is not None:
                pl.positions.extend(pos)
        doc_meta.extend(DocMeta(offset + i, p, t) for i, (p, t) in enumerate(part.doc_meta))
        doc_lengths.extend(part.doc_lengths)
        if texts is not None and part.texts is not None:
            texts.extend(part.texts)
        else:
            texts = None
    ordered = {t: postings[t] for t in sorted(postings)}
    return Index(ordered, doc_meta, doc_lengths, texts, has_positions=has_positions)


def chunked(paths: Sequence[Path], n_chunks: int) -> list[list[Path]]:
    """Split into `n_chunks` contiguous, nearly equal slices (order preserved)."""
    n_chunks = max(1, min(n_chunks, len(paths)))
    size, extra = divmod(len(paths), n_chunks)
    out: list[list[Path]] = []
    start = 0
    for i in range(n_chunks):
        end = start + size + (1 if i < extra else 0)
        out.append(list(paths[start:end]))
        start = end
    return out


def _make_executor(name: ExecutorName, workers: int) -> Executor | None:
    if name == "threads":
        return ThreadPoolExecutor(max_workers=workers)
    if name == "processes":
        return ProcessPoolExecutor(max_workers=workers)
    return None


@dataclass
class BuildReport:
    executor: ExecutorName
    workers: int
    chunks: int
    wall: float = 0.0
    build: float = 0.0
    merge: float = 0.0
    cpu: float = 0.0  # parent process CPU + (processes) every worker process's CPU
    partial_cpu: list[float] = field(default_factory=lambda: list[float]())


def build_parallel(
    corpus: Path | str,
    *,
    workers: int = 4,
    executor: ExecutorName = "processes",
    positions: bool = True,
    store_text: bool = True,
    chunks_per_worker: int = 4,
    limit: int | None = None,
    on_chunk: Callable[[int], None] | None = None,
) -> tuple[Index, BuildReport]:
    """Index `corpus` with partial builds on an executor, then a serial merge.

    A worker exception propagates out of ``map`` with the worker's traceback
    attached (``concurrent.futures`` chains it as ``__cause__``), and the pool
    cancels whatever hasn't started — the build fails loudly instead of hanging.
    """
    paths = iter_paths(corpus)
    if limit is not None:
        paths = paths[:limit]  # .txt corpora: one path = one document
    n_chunks = 1 if executor == "serial" else workers * chunks_per_worker
    chunks = chunked(paths, n_chunks)
    report = BuildReport(executor, workers, len(chunks))
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    pool = _make_executor(executor, workers)
    partials: list[PartialIndex] = []
    try:
        jobs: Iterator[PartialIndex]
        if pool is None:
            jobs = (build_partial(c, positions, store_text) for c in chunks)
        else:
            jobs = pool.map(
                build_partial, chunks, [positions] * len(chunks), [store_text] * len(chunks)
            )
        for part in jobs:  # results arrive in chunk order
            partials.append(part)
            if on_chunk is not None:
                on_chunk(len(part.doc_meta))
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    t1 = time.perf_counter()
    index = merge(partials)
    t2 = time.perf_counter()
    report.build, report.merge, report.wall = t1 - t0, t2 - t1, t2 - t0
    report.partial_cpu = [p.cpu_seconds for p in partials]
    report.cpu = time.process_time() - cpu0  # includes every thread of this process
    if executor == "processes":
        per_worker: dict[int, float] = {}
        for p in partials:
            per_worker[p.worker_pid] = max(per_worker.get(p.worker_pid, 0.0), p.worker_cpu_total)
        report.cpu += sum(per_worker.values())
    log.info(
        "built %r with %s×%d in %.2fs (merge %.2fs)",
        index,
        executor,
        workers,
        report.wall,
        report.merge,
    )
    return index, report
