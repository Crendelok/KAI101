"""The ``findex`` command line (Lab 4).

Logging is configured once, here; every diagnostic goes to stderr so stdout
stays clean for results (and for ``--json`` piping).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterator
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex import __version__
from findex.corpus import Document, iter_documents, iter_paths
from findex.index import Index, build_index
from findex.query import QueryError
from findex.scoring import get_scorer
from findex.search import search, snippet_for

log = logging.getLogger("findex")
out = Console()
err = Console(stderr=True)

app = typer.Typer(
    name="findex",
    help="A full-text search engine over a corpus you choose.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class ExecutorChoice(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(console=err, show_time=False, show_path=False)],
        force=True,
    )


def _fail(message: str, code: int = 1) -> typer.Exit:
    err.print(f"[red]error:[/] {message}")
    return typer.Exit(code)


def _version(value: bool) -> None:
    if value:
        out.print(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v info, -vv debug (to stderr).")
    ] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    _configure_logging(verbose)


def _load(path: Path) -> Index:
    from findex.store import load

    if not path.exists():
        raise _fail(f"index file not found: {path}")
    try:
        return load(path, lazy=True)
    except ValueError as exc:
        raise _fail(str(exc)) from exc


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Corpus directory or .txt/.jsonl file.")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Where to write the index.")] = Path(
        "index.fidx"
    ),
    positions: Annotated[bool, typer.Option(help="Record positions (phrase queries).")] = True,
    limit: Annotated[int | None, typer.Option(help="Only index the first N documents.")] = None,
    executor: Annotated[
        ExecutorChoice, typer.Option(help="How to parallelize the build.")
    ] = ExecutorChoice.processes,
    workers: Annotated[
        int, typer.Option(min=1, help="Worker threads/processes (default: min(8, CPU count)).")
    ] = min(8, os.cpu_count() or 1),
) -> None:
    """Build an index from a corpus and save it."""
    from findex.parallel import build_parallel
    from findex.store import save

    if not corpus.exists():
        raise _fail(f"corpus not found: {corpus}")
    paths = iter_paths(corpus)
    if not paths:
        raise _fail(f"no .txt/.jsonl files under {corpus}")
    total = limit if limit is not None else None
    if total is None and all(p.suffix == ".txt" for p in paths):
        total = len(paths)
    if executor is not ExecutorChoice.serial:
        with Progress(
            TextColumn(f"[bold]indexing ({executor.value} × {workers})"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err,
            transient=True,
        ) as progress:
            task = progress.add_task("docs", total=total)
            index, report = build_parallel(
                corpus,
                workers=workers,
                executor=executor.value,
                positions=positions,
                limit=limit,
                on_chunk=lambda n: progress.advance(task, n),
            )
        log.info("build %.2fs + merge %.2fs, cpu %.1fs", report.build, report.merge, report.cpu)
        save(index, out_path)
        err.print(f"[green]✓[/] {index!r} → {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB)")
        return
    with Progress(
        TextColumn("[bold]indexing"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
        transient=True,
    ) as progress:
        task = progress.add_task("docs", total=total)

        def tracked(docs: Iterator[Document]) -> Iterator[Document]:
            for doc in docs:
                yield doc
                progress.advance(task)

        docs = iter_documents(corpus)
        if limit is not None:
            docs = islice(docs, limit)
        index = build_index(tracked(docs), positions=positions)
    save(index, out_path)
    err.print(f"[green]✓[/] {index!r} → {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB)")


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(help="Index file.")],
    query: Annotated[str, typer.Argument(help='Query: words, OR, NOT, ( ), "phrases".')],
    k: Annotated[int, typer.Option("--k", "-k", min=1, max=1000, help="Results to show.")] = 10,
    scorer: Annotated[ScorerChoice, typer.Option(help="Ranking function.")] = ScorerChoice.bm25,
    as_json: Annotated[bool, typer.Option("--json", help="JSON lines on stdout.")] = False,
) -> None:
    """Search an index and print ranked results with snippets."""
    index = _load(index_path)
    try:
        results = search(index, query, get_scorer(scorer.value), k)
    except QueryError as exc:
        raise _fail(f"bad query: {exc}", code=2) from exc
    if as_json:
        for rank, r in enumerate(results, start=1):
            snip = snippet_for(index, r.doc_id, results.terms)
            record = {
                "rank": rank,
                "doc_id": r.doc_id,
                "score": round(r.score, 4),
                "title": r.title,
                "path": index.doc_meta[r.doc_id].path,
                "snippet": snip.text,
            }
            sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    table = Table(title=f"{results.total:,} matches for {query!r}", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("document")
    for rank, r in enumerate(results, start=1):
        snip = snippet_for(index, r.doc_id, results.terms)
        table.add_row(str(rank), f"{r.score:.2f}", f"[bold]{r.title}[/]\n{snip.rich()}")
    out.print(table)


@app.command("stats")
def stats_cmd(
    index_path: Annotated[Path, typer.Argument(help="Index file.")],
    top: Annotated[int, typer.Option(help="How many top terms to list.")] = 20,
) -> None:
    """Document count, tokens, vocabulary and the most frequent terms."""
    index = _load(index_path)
    cf: Counter[str] = Counter()
    for term in index:
        cf[term] = sum(index[term].tfs)
    summary = Table(show_header=False, box=None)
    summary.add_row("documents", f"{index.num_docs:,}")
    summary.add_row("total tokens", f"{index.total_tokens:,}")
    summary.add_row("vocabulary", f"{len(index):,}")
    summary.add_row("avg doc length", f"{index.avg_doc_length:,.1f} tokens")
    summary.add_row("positions", "yes" if index.has_positions else "no")
    summary.add_row("index file", f"{index_path.stat().st_size / 2**20:.1f} MiB")
    out.print(summary)
    terms = Table(title=f"top {top} terms")
    terms.add_column("term")
    terms.add_column("count", justify="right")
    terms.add_column("df", justify="right")
    for term, n in cf.most_common(top):
        terms.add_row(term, f"{n:,}", f"{index.df(term):,}")
    out.print(terms)




if __name__ == "__main__":
    app()
