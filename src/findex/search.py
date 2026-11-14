"""Boolean search (Lab 2).

Query syntax: words are ANDed; ``OR`` and ``NOT`` are operators; evaluation is
strictly left to right — ``a b OR c NOT d`` means ``((a AND b) OR c) AND NOT d``.

    python -m findex.search index.fidx "whale ship NOT boat" [--engine merge|set]
"""

from __future__ import annotations

import argparse
import logging
from array import array
from collections.abc import Sequence
from pathlib import Path

from findex.index import Index
from findex.setops import Engine, difference, intersect, union
from findex.tokenize import tokenize


def term_doc_ids(index: Index, word: str, engine: Engine) -> Sequence[int]:
    """Doc ids for one query word (a word like ``state-of-the-art`` ANDs its tokens)."""
    result: Sequence[int] | None = None
    for tok in tokenize(word):
        ids = index[tok].doc_ids if tok in index else array("I")
        result = ids if result is None else intersect(result, ids, engine)
    return result if result is not None else array("I")


def boolean_search(index: Index, query: str, engine: Engine = "merge") -> Sequence[int]:
    result: Sequence[int] | None = None
    op = "AND"
    for word in query.split():
        if word in ("OR", "NOT", "AND"):
            op = word
            continue
        ids = term_doc_ids(index, word, engine)
        if result is None:
            result = difference(index.all_doc_ids(), ids, engine) if op == "NOT" else ids
        elif op == "OR":
            result = union(result, ids, engine)
        elif op == "NOT":
            result = difference(result, ids, engine)
        else:
            result = intersect(result, ids, engine)
        op = "AND"
    return result if result is not None else array("I")


def main(argv: list[str] | None = None) -> int:
    from findex.measure import measured
    from findex.store import load

    p = argparse.ArgumentParser(prog="python -m findex.search")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--engine", choices=["merge", "set"], default="merge")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with measured("load") as m_load:
        index = load(args.index)
    with measured("search") as m_search:
        ids = boolean_search(index, args.query, args.engine)
    print(f"{len(ids):,} matching documents")
    for doc_id in ids[: args.k]:
        print(f"  [{doc_id}] {index.title(doc_id)}")
    print(f"load:   {m_load}")
    print(f"search: {m_search}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
