from __future__ import annotations

import math
import random

import numpy as np
import pytest

from findex.corpus import Document
from findex.fast import search_fast, top_k
from findex.index import Index, build_index
from findex.scoring import BM25, TfIdf
from findex.search import search

QUERIES = [
    "whale",
    "the",
    "white whale",
    "whale OR cat OR sea",
    '"white whale" OR ship',
    "whale NOT sea",
    "NOT whale",
    "zebra",
    "the OR a OR of",
    "кит",
]


@pytest.mark.parametrize("scorer", [BM25(), TfIdf(), BM25(k1=0.9, b=0.0)])
@pytest.mark.parametrize("query", QUERIES)
def test_numpy_ranking_is_identical(index: Index, query: str, scorer: BM25 | TfIdf):
    slow = search(index, query, scorer, k=8)
    fast = search_fast(index, query, scorer, k=8)
    assert [r.doc_id for r in fast] == [r.doc_id for r in slow]
    assert all(
        math.isclose(a.score, b.score, rel_tol=1e-12) for a, b in zip(fast, slow, strict=True)
    )
    assert fast.total == slow.total and fast.terms == slow.terms


@pytest.fixture(scope="module")
def big_index() -> Index:
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(3000)]
    weights = [1 / (i + 1) for i in range(len(vocab))]  # Zipf-ish, like real text
    docs = [
        Document(i, f"d{i}", f"d{i}", " ".join(rng.choices(vocab, weights, k=rng.randint(20, 400))))
        for i in range(1500)
    ]
    return build_index(docs, store_text=False)


@pytest.mark.parametrize("query", ["w0", "w2999 OR w1500", "w0 OR w1 OR w2", "w3 w10"])
def test_identical_on_a_bigger_random_corpus(big_index: Index, query: str):
    for offset in (0, 7):
        slow = search(big_index, query, BM25(), k=20, offset=offset)
        fast = search_fast(big_index, query, BM25(), k=20, offset=offset)
        assert [r.doc_id for r in fast] == [r.doc_id for r in slow]


def test_top_k_keeps_ties_in_doc_order():
    ids = np.array([5, 3, 9, 1, 7], dtype=np.int64)
    scores = np.array([1.0, 2.0, 2.0, 2.0, 0.5])
    assert top_k(ids, scores, 2) == [(1, 2.0), (3, 2.0)]
    assert top_k(ids, scores, 10)[-1] == (7, 0.5)
    assert top_k(ids[:0], scores[:0], 3) == []
