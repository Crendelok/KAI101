from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import DocMeta, Index, Posting, PostingList
from findex.search import boolean_search
from findex.store import load, open_index, save

from .conftest import doc_named


def test_posting_and_docmeta_are_hashable_values():
    a, b = Posting(1, 2, (3, 4)), Posting(1, 2, (3, 4))
    assert a == b and len({a, b}) == 1
    assert len({DocMeta(0, "p", "t"), DocMeta(0, "p", "t")}) == 1
    with pytest.raises(AttributeError):
        a.tf = 5  # type: ignore[misc]  # frozen: assignment must fail


def test_postings_sorted_and_tf(index: Index):
    for term in index:
        ids = list(index[term].doc_ids)
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
    repeat = doc_named(index, "05-repeat")
    pl = index["whale"]
    assert pl[pl.find(repeat)].tf == 20


def test_positions(index: Index):
    pl = index["white"]
    d = doc_named(index, "06-phrase")
    # "phrase the sea was white and the white whale was not"
    assert list(pl.positions_at(pl.find(d))) == [4, 7]


def test_mapping_protocol(index: Index):
    assert "whale" in index and "zebra" not in index
    assert len(index) == len(list(index))
    with pytest.raises(KeyError):
        index["zebra"]
    assert index.df("whale") == 4 and index.df("zebra") == 0
    assert repr(index).startswith("<Index docs=8")
    assert index.text(doc_named(index, "02-cat")).startswith("The Cat")


def test_posting_list_rejects_out_of_order():
    pl = PostingList()
    pl.append(3, 1)
    with pytest.raises(ValueError):
        pl.append(2, 1)


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_search(index: Index, engine: str):
    whale = set(boolean_search(index, "whale", engine))  # type: ignore[arg-type]
    sea = set(boolean_search(index, "sea", engine))  # type: ignore[arg-type]
    assert set(boolean_search(index, "whale sea", engine)) == whale & sea  # type: ignore[arg-type]
    assert set(boolean_search(index, "whale OR sea", engine)) == whale | sea  # type: ignore[arg-type]
    assert set(boolean_search(index, "whale NOT sea", engine)) == whale - sea  # type: ignore[arg-type]
    everything = set(range(index.num_docs))
    assert set(boolean_search(index, "NOT whale", engine)) == everything - whale  # type: ignore[arg-type]
    assert list(boolean_search(index, "zebra", engine)) == []  # type: ignore[arg-type]


def _same(a: Index, b: Index) -> None:
    assert a.num_docs == b.num_docs and set(a) == set(b)
    for term in a:
        assert list(a[term]) == list(b[term])
    assert list(a.doc_lengths) == list(b.doc_lengths)
    assert list(a.doc_meta) == list(b.doc_meta)
    assert [a.text(i) for i in range(a.num_docs)] == [b.text(i) for i in range(b.num_docs)]


@pytest.mark.parametrize("name", ["ix.fidx", "ix.pkl"])
def test_save_load_round_trip(index: Index, tmp_path: Path, name: str):
    save(index, tmp_path / name)
    _same(index, load(tmp_path / name))


def test_lazy_mmap_load_and_context_manager(index: Index, tmp_path: Path):
    save(index, tmp_path / "ix.fidx")
    with open_index(tmp_path / "ix.fidx") as ix:
        _same(index, ix)
    with pytest.raises(RuntimeError), open_index(tmp_path / "ix.fidx") as ix:
        raise RuntimeError("body fails; mmap must still be closed")


def test_rejects_non_index_file(tmp_path: Path):
    bad = tmp_path / "bad.fidx"
    bad.write_bytes(b"definitely not an index")
    with pytest.raises(ValueError):
        load(bad)
