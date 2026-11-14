"""Search: Lab 2's boolean evaluation and Lab 3's ranked search.

python -m findex.search index.fidx 'whale "white whale" NOT ship' [--scorer bm25|tfidf]
"""

from __future__ import annotations

import argparse
import functools
import heapq
import logging
from array import array
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload, override

from findex.index import Index, Posting
from findex.query import parse
from findex.scoring import BM25, Scorer
from findex.setops import Engine, difference, intersect, union
from findex.snippets import Snippet, make_snippet
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

# --- Lab 2: strictly left-to-right boolean evaluation ----------------------------


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


# --- Lab 3: ranked search -------------------------------------------------------------


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    """Sorts best-first: ``sorted(results)`` orders by score desc, then doc id."""

    sort_key: tuple[float, int] = field(init=False, repr=False)
    doc_id: int = field(compare=False)
    score: float = field(compare=False)
    title: str = field(compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_key", (-self.score, self.doc_id))


class ResultList(Sequence[SearchResult]):
    """Top-k hits plus ``total`` (how many documents matched before cutting to k).

    Slicing returns another ResultList, so ``results[:5]`` keeps ``total``.
    """

    def __init__(self, hits: Sequence[SearchResult], total: int, terms: tuple[str, ...]) -> None:
        self._hits = list(hits)
        self.total = total
        self.terms = terms

    @overload
    def __getitem__(self, i: int) -> SearchResult: ...
    @overload
    def __getitem__(self, i: slice) -> ResultList: ...
    @override
    def __getitem__(self, i: int | slice) -> SearchResult | ResultList:
        if isinstance(i, slice):
            return ResultList(self._hits[i], self.total, self.terms)
        return self._hits[i]

    @override
    def __len__(self) -> int:
        return len(self._hits)

    @override
    def __iter__(self) -> Iterator[SearchResult]:
        return iter(self._hits)

    @override
    def __repr__(self) -> str:
        return f"ResultList({len(self)} of {self.total})"


@functools.lru_cache(maxsize=256)
def _matching_ids(index: Index, query: str, engine: Engine) -> tuple[array[int], tuple[str, ...]]:
    node = parse(query)
    ids = node.evaluate(index, engine)
    terms = tuple(dict.fromkeys(t for t in node.positive_terms() if t in index))
    return (ids if isinstance(ids, array) else array("I", ids)), terms


@timed
def matching_ids(
    index: Index, query: str, engine: Engine = "merge"
) -> tuple[array[int], tuple[str, ...]]:
    """query -> (matching doc ids, ranking terms). LRU-cached: a repeat is ~0 ms."""
    before = _matching_ids.cache_info().hits
    result = _matching_ids(index, query, engine)
    if _matching_ids.cache_info().hits > before:
        log.debug("query cache hit for %r", query)
    return result


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    *,
    offset: int = 0,
    engine: Engine = "merge",
) -> ResultList:
    """Match with the query tree, score matches term by term, keep the top k."""
    scorer = scorer or BM25()
    ids, terms = matching_ids(index, query, engine)
    if not ids:
        return ResultList([], 0, terms)
    matched = set(ids)
    scores: defaultdict[int, float] = defaultdict(float)
    for term in terms:
        pl = index[term]
        for doc_id, tf in zip(pl.doc_ids, pl.tfs, strict=True):
            if doc_id in matched:
                scores[doc_id] += scorer.score(term, Posting(doc_id, tf), index)
    if not terms:  # e.g. "NOT whale": everything matches equally
        scores = defaultdict(float, dict.fromkeys(ids[: offset + k], 0.0))
    top = heapq.nlargest(offset + k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
    hits = [SearchResult(doc_id=d, score=s, title=index.title(d)) for d, s in top[offset:]]
    return ResultList(hits, len(ids), terms)


def snippet_for(index: Index, doc_id: int, terms: Sequence[str]) -> Snippet:
    text = index.text(doc_id)
    return make_snippet(text, terms) if text else Snippet("")


def main(argv: list[str] | None = None) -> int:
    from findex.scoring import get_scorer
    from findex.store import open_index

    p = argparse.ArgumentParser(prog="python -m findex.search")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    p.add_argument("--engine", choices=["merge", "set"], default="merge")
    p.add_argument("--boolean", action="store_true", help="Lab 2 left-to-right boolean mode")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--repeat", type=int, default=1, help="run the query N times (cache demo)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    with open_index(args.index) as index:
        if args.boolean:
            ids = boolean_search(index, args.query, args.engine)
            print(f"{len(ids):,} matching documents")
            for doc_id in ids[: args.k]:
                print(f"  [{doc_id}] {index.title(doc_id)}")
            return 0
        results = ResultList([], 0, ())
        for _ in range(args.repeat):
            results = search(index, args.query, get_scorer(args.scorer), args.k, engine=args.engine)
        print(f"{results.total:,} matching documents\n")
        for rank, r in enumerate(results, start=1):
            print(f"{rank:>3}. {r.score:7.3f}  {r.title}")
            print(f"        {snippet_for(index, r.doc_id, results.terms).plain()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
