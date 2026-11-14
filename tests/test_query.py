from __future__ import annotations

import pytest

from findex.index import Index
from findex.query import And, Not, Or, Phrase, QueryError, Term, parse

from .conftest import doc_named

a, b, c = Term("a"), Term("b"), Term("c")


@pytest.mark.parametrize(
    ("query", "tree"),
    [
        ("a", a),
        ("a b", And(a, b)),
        ("a AND b", And(a, b)),
        ("a b c", And(And(a, b), c)),
        ("a OR b c", Or(a, And(b, c))),
        ("a b OR c", Or(And(a, b), c)),
        ("(a OR b) c", And(Or(a, b), c)),
        ("NOT a", Not(a)),
        ("a NOT b", And(a, Not(b))),
        ("NOT NOT a", Not(Not(a))),
        ('"a b" c', And(Phrase(("a", "b")), c)),
        ("A", a),  # casefolded
        ("state-of-the-art", Phrase(("state", "of", "the", "art"))),
        ("a , b", And(a, b)),  # punctuation-only words vanish
    ],
)
def test_parse_trees(query: str, tree: object):
    assert parse(query) == tree


def test_operators_build_trees():
    assert (a & b) | ~c == Or(And(a, b), Not(c))


@pytest.mark.parametrize("bad", ["", "   ", "(a", "a)", "a OR", "OR a", "NOT", ","])
def test_bad_queries_raise(bad: str):
    with pytest.raises(QueryError):
        parse(bad)


def ids(index: Index, q: str) -> set[int]:
    return set(parse(q).evaluate(index))


def test_evaluate_matches_set_algebra(index: Index):
    whale, sea = ids(index, "whale"), ids(index, "sea")
    assert ids(index, "whale sea") == whale & sea
    assert ids(index, "whale OR sea") == whale | sea
    assert ids(index, "whale NOT sea") == whale - sea
    assert ids(index, "NOT whale") == set(range(index.num_docs)) - whale
    assert ids(index, "(whale OR cat) NOT sea") == (whale | ids(index, "cat")) - sea


def test_phrase_needs_adjacency(index: Index):
    assert ids(index, '"white whale"') == {
        doc_named(index, "00-whale"),
        doc_named(index, "06-phrase"),
    }
    assert ids(index, '"whale white"') == set()
    assert ids(index, '"кит пливе"') == {doc_named(index, "07-ukr")}


def test_phrase_without_positions_is_an_error(index_nopos: Index):
    with pytest.raises(QueryError):
        parse('"white whale"').evaluate(index_nopos)
