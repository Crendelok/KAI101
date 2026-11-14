"""Vectorized ranking (Lab 8): BM25 / TF-IDF as NumPy array expressions.

The Lab 3 scorer calls a Python method once per (term, posting) and adds into
a dict: ~1 µs of interpreter overhead per posting. Here each term is one
handful of array operations over its whole posting list, accumulated into a
dense ``float64[N]`` score vector; top-k is ``np.argpartition`` (O(n)) on the
matched documents only.

Rankings are *identical* to ``findex.search.search`` (tested): the same
float64 operations in the same order, and the same tie-break (score desc,
then doc id asc).
"""

from __future__ import annotations

import math
import weakref

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.scoring import BM25, TfIdf
from findex.search import ResultList, SearchResult, matching_ids
from findex.setops import Engine
from findex.timing import timed

type F64 = npt.NDArray[np.float64]


class ArrayView:
    """Zero-copy NumPy views over an Index's postings, plus per-document arrays."""

    def __init__(self, index: Index) -> None:
        self.index = index
        self.n = index.num_docs
        self.doc_len: F64 = np.asarray(index.doc_lengths, dtype=np.float64)
        self.avg = index.avg_doc_length

    def postings(self, term: str) -> tuple[npt.NDArray[np.uint32], F64]:
        pl = self.index[term]
        ids = np.frombuffer(pl.doc_ids, dtype=np.uint32)  # no copy: views the array's buffer
        tfs = np.frombuffer(pl.tfs, dtype=np.uint32).astype(np.float64)
        return ids, tfs


_views: weakref.WeakKeyDictionary[Index, ArrayView] = weakref.WeakKeyDictionary()


def view(index: Index) -> ArrayView:
    """Per-index arrays, built once and dropped when the index is garbage-collected."""
    v = _views.get(index)
    if v is None:
        v = _views[index] = ArrayView(index)
    return v


def term_scores(
    v: ArrayView, term: str, scorer: BM25 | TfIdf
) -> tuple[npt.NDArray[np.uint32], F64]:
    ids, tf = v.postings(term)
    df = len(ids)
    if isinstance(scorer, BM25):
        idf = scorer.idf(df, v.n)
        norm = 1.0 - scorer.b + scorer.b * (v.doc_len[ids] / v.avg if v.avg else 1.0)
        tfw = tf * (scorer.k1 + 1.0) / (tf + scorer.k1 * norm)
        return ids, tfw * idf
    idf = math.log(v.n / df) if df else 0.0
    return ids, (1.0 + np.log(tf)) * idf


def top_k(doc_ids: npt.NDArray[np.int64], scores: F64, k: int) -> list[tuple[int, float]]:
    """Best k by (score desc, doc id asc), using argpartition for the O(n) cut."""
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    if k < n:
        cut = np.argpartition(-scores, k - 1)[:k]
        threshold = scores[cut].min()
        cand = np.flatnonzero(scores >= threshold)  # keep every tie at the boundary
    else:
        cand = np.arange(n)
    order = np.lexsort((doc_ids[cand], -scores[cand]))[:k]
    pick = cand[order]
    return [(int(d), float(s)) for d, s in zip(doc_ids[pick], scores[pick], strict=True)]


@timed
def search_fast(
    index: Index,
    query: str,
    scorer: BM25 | TfIdf | None = None,
    k: int = 10,
    *,
    offset: int = 0,
    engine: Engine = "merge",
) -> ResultList:
    """Drop-in replacement for ``findex.search.search`` with vectorized scoring."""
    scorer = scorer or BM25()
    ids, terms = matching_ids(index, query, engine)
    if not ids:
        return ResultList([], 0, terms)
    v = view(index)
    matched = np.frombuffer(ids, dtype=np.uint32).astype(np.int64)
    if not terms:
        top = [(int(d), 0.0) for d in matched[: offset + k]]
    else:
        acc = np.zeros(v.n, dtype=np.float64)
        for term in terms:
            t_ids, t_scores = term_scores(v, term, scorer)
            acc[t_ids] += t_scores  # ids are unique within a posting list: no np.add.at needed
        top = top_k(matched, acc[matched], offset + k)
    hits = [SearchResult(doc_id=d, score=s, title=index.title(d)) for d, s in top[offset:]]
    return ResultList(hits, len(matched), terms)
