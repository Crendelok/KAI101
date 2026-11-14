"""Semantic/hybrid plumbing, tested with a fake encoder (no model download, no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from findex.index import Index
from findex.search import snippet_for
from findex.semantic import SemanticIndex, chunks, rrf, search_mode

from .conftest import doc_named


def test_chunks_cover_text_with_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    out = list(chunks(text, words=200, overlap=40))
    assert [len(c.split()) for _, c in out] == [200, 200, 180]
    assert out[1][1].split()[0] == "w160"  # 40 words of overlap
    assert text[out[1][0] :].startswith("w160")  # offsets point into the original
    assert list(chunks("", 10, 2)) == []
    assert list(chunks("one two", 200, 40)) == [(0, "one two")]


def test_rrf_rewards_agreement():
    fused = rrf([1, 2, 3], [3, 1, 4])
    assert [d for d, _ in fused][:2] == [1, 3]  # in both lists, near the top
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)[4] < dict(fused)[2]


class FakeSemantic(SemanticIndex):
    """Doc d's chunk vector is the unit vector e_d; the query 'about d' encodes to e_d."""

    def encode(self, query: str) -> np.ndarray:  # type: ignore[override]
        v = np.zeros(self.embeddings.shape[1], dtype=np.float32)
        v[int(query.split()[-1])] = 1.0
        return v


@pytest.fixture
def fake(index: Index) -> FakeSemantic:
    n = index.num_docs
    emb = np.eye(n, dtype=np.float32)
    return FakeSemantic(emb, np.arange(n, dtype=np.int64), np.full(n, 5, dtype=np.int64), "fake")


def test_semantic_mode_ranks_by_cosine(index: Index, fake: FakeSemantic):
    cat = doc_named(index, "02-cat")
    res = search_mode(index, fake, f"about {cat}", mode="semantic", k=3)
    assert res[0].doc_id == cat and res[0].score == pytest.approx(1.0)
    assert res.anchors[cat] == 5
    assert snippet_for(index, cat, (), res.anchors[cat]).text.startswith("…")


def test_hybrid_fuses_keyword_and_semantic(index: Index, fake: FakeSemantic):
    whale = doc_named(index, "00-whale")
    res = search_mode(index, fake, f"whale {whale}", mode="hybrid", k=5)
    assert res[0].doc_id == whale  # first in both rankings
    keyword = search_mode(index, fake, "whale", mode="keyword", k=5)
    assert keyword.total == 4
