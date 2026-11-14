from __future__ import annotations

import logging

import pytest

from findex.index import Index, Posting
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import SearchResult, _matching_ids, search, snippet_for
from findex.snippets import make_snippet

from .conftest import doc_named


def top(index: Index, q: str, scorer: Scorer | None = None) -> list[int]:
    return [r.doc_id for r in search(index, q, scorer or BM25())]


# --- the three Lab 3 sanity checks ------------------------------------------------


def test_rare_term_outranks_common_term(index: Index):
    # "narwhal" is in 1 doc, "whale" in 4 — including one that says "whale" 20 times.
    assert top(index, "narwhal OR whale")[0] == doc_named(index, "04-rare")


def test_bm25_saturates_repetition():
    bm25 = BM25()
    w = [bm25.tf_weight(tf, 100, 100.0) for tf in range(1, 21)]
    assert w[19] - w[18] < 0.01 * w[0]  # the 20th "whale" adds < 1% of the first
    assert all(x < y for x, y in zip(w, w[1:], strict=False))  # but never hurts
    tfidf = TfIdf()
    assert tfidf.tf_weight(20, 0, 0) - tfidf.tf_weight(19, 0, 0) > 0.04  # tf-idf keeps going


def test_short_doc_beats_long_doc_with_one_match(index: Index):
    short, long_ = doc_named(index, "06-phrase"), doc_named(index, "03-long")
    s = {r.doc_id: r.score for r in search(index, "whale", BM25(), k=10)}
    assert s[short] > s[long_]


# --- results, scorers, cache, snippets -----------------------------------------------


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_scorers_satisfy_protocol(index: Index, scorer: Scorer):
    assert scorer.score("whale", Posting(0, 1), index) > 0


def test_results_sorted_and_total(index: Index):
    res = search(index, "whale OR sea", k=3)
    expected_total = len(set(index["whale"].doc_ids) | set(index["sea"].doc_ids))
    assert len(res) == 3 and res.total == expected_total
    assert list(res) == sorted(res)
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)
    assert res[:2].total == res.total and len(res[:2]) == 2


def test_search_result_orders_best_first():
    lo, hi = (
        SearchResult(doc_id=1, score=1.0, title="x"),
        SearchResult(doc_id=2, score=2.0, title="y"),
    )
    assert sorted([lo, hi]) == [hi, lo]


def test_offset_paginates(index: Index):
    full = search(index, "the", k=6)
    page2 = search(index, "the", k=3, offset=3)
    assert [r.doc_id for r in page2] == [r.doc_id for r in full][3:6]


def test_no_match_and_pure_not(index: Index):
    assert search(index, "zebra").total == 0
    assert search(index, "NOT whale").total == index.num_docs - 4


def test_query_cache_hit_is_logged(index: Index, caplog: pytest.LogCaptureFixture):
    _matching_ids.cache_clear()
    with caplog.at_level(logging.DEBUG):
        search(index, "ship sea")
        search(index, "ship sea")
    assert _matching_ids.cache_info().hits == 1
    assert "query cache hit" in caplog.text
    assert "search took" in caplog.text


def test_snippet_highlights_terms(index: Index):
    snip = snippet_for(index, doc_named(index, "00-whale"), ["whale", "ahab"])
    assert "«Ahab»" in snip.plain() and "«whale»" in snip.plain()
    assert "<mark>whale</mark>" in snip.html()


def test_snippet_window_and_escape():
    text = "x " * 200 + "needle <b> " + "y " * 200
    snip = make_snippet(text, {"needle"})
    assert snip.text.startswith("…") and snip.text.endswith("…")
    assert len(snip.text) < 200
    assert "<mark>needle</mark> &lt;b&gt;" in snip.html()
    assert make_snippet("", {"a"}).text == ""
    assert make_snippet("no match here", {"zzz"}).spans == ()
