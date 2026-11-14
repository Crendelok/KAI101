"""Hypothesis properties: claims about *all* inputs, not a handful of examples."""

from __future__ import annotations

import string
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.setops import difference_merge, intersect_merge, union_merge
from findex.store import load, save
from findex.tokenize import tokenize

sorted_ids = st.lists(st.integers(0, 500), max_size=60, unique=True).map(sorted)


@given(sorted_ids, sorted_ids)
def test_merge_equals_set_algebra(a: list[int], b: list[int]):
    assert list(intersect_merge(a, b)) == sorted(set(a) & set(b))
    assert list(union_merge(a, b)) == sorted(set(a) | set(b))
    assert list(difference_merge(a, b)) == sorted(set(a) - set(b))


@given(st.text())
def test_tokens_are_nonempty_words(text: str):
    for tok in tokenize(text):
        assert tok and not any(ch.isspace() for ch in tok)


ascii_text = st.text(alphabet=string.ascii_letters + string.digits + " .,;:-'!?\n")


@given(ascii_text)
def test_tokenize_ignores_case_and_is_stable(text: str):
    toks = list(tokenize(text))
    assert list(tokenize(text.upper())) == toks
    assert list(tokenize(" ".join(toks))) == toks


words = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6).filter(
    lambda w: w not in {"and", "or", "not"}
)
leaves = st.one_of(
    words.map(Term),
    st.lists(words, min_size=2, max_size=3).map(lambda ws: Phrase(tuple(ws))),
)
trees: st.SearchStrategy[Node] = st.recursive(
    leaves,
    lambda kids: st.one_of(
        st.tuples(kids, kids).map(lambda t: And(*t)),
        st.tuples(kids, kids).map(lambda t: Or(*t)),
        kids.map(Not),
    ),
    max_leaves=8,
)


@given(trees)
def test_query_tree_round_trips_through_text(tree: Node):
    assert parse(str(tree)) == tree


corpora = st.lists(st.text(alphabet=string.ascii_lowercase + " é\n", max_size=80), max_size=8)


@settings(max_examples=40, deadline=None)
@given(corpora)
def test_index_invariants_and_round_trip(texts: list[str]):
    docs = [Document(i, f"d{i}", f"t{i}", t) for i, t in enumerate(texts)]
    index = build_index(docs, positions=True)
    for i, text in enumerate(texts):
        assert index.doc_length(i) == len(list(tokenize(text)))
    for term in index:
        pl = index[term]
        assert list(pl.doc_ids) == sorted(set(pl.doc_ids))
        assert sum(pl.tfs) == len(pl.positions or ())
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ix.fidx"
        save(index, path)
        again = load(path)
        assert {t: list(again[t]) for t in again} == {t: list(index[t]) for t in index}
        assert [again.text(i) for i in range(len(texts))] == texts
