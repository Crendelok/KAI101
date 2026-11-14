"""Semantic and hybrid search (Lab 8).

* **Chunk** every document into ~200-word windows (with overlap): an embedding
  of a whole 1,500-word section blurs into an average of everything in it.
* **Embed** chunks with a sentence-transformer, L2-normalize, store
  ``E: float32[n_chunks, d]`` + chunk -> (doc id, char offset) as ``.npy``.
* **Search**: embed the query (normalized), ``E @ q`` gives cosine similarity
  for every chunk at once; keep the best chunk per document.
* **Hybrid**: Reciprocal Rank Fusion of the keyword and semantic rankings,
  ``score(d) = Σ 1 / (60 + rank_i(d))`` — rank-based, so BM25's unbounded
  scores and cosine's [-1, 1] never need to be put on one scale.

Requires the ``semantic`` extra: ``uv sync --extra semantic``.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.query import QueryError
from findex.scoring import get_scorer
from findex.search import ResultList, SearchResult
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
CANDIDATES = 100
_WORD = re.compile(r"\S+")


def chunks(text: str, words: int = 200, overlap: int = 40) -> Iterator[tuple[int, str]]:
    """Yield ``(char_offset, chunk_text)`` windows of `words` words, overlapping by `overlap`."""
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not spans:
        return
    step = max(1, words - overlap)
    for i in range(0, len(spans), step):
        window = spans[i : i + words]
        yield window[0][0], text[window[0][0] : window[-1][1]]
        if i + words >= len(spans):
            break


def emb_paths(index_path: Path) -> tuple[Path, Path, Path]:
    base = index_path.with_suffix("")
    return (
        base.with_suffix(".emb.npy"),
        base.with_suffix(".emb.chunks.npy"),
        base.with_suffix(".emb.json"),
    )


class Encoder(Protocol):
    """The slice of sentence-transformers' API findex uses."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


def _model(name: str) -> Encoder:
    # Imported dynamically: sentence-transformers (and torch) is an optional extra,
    # and type-checking/CI must work without it installed.
    module = importlib.import_module("sentence_transformers")
    cls: Any = module.SentenceTransformer
    return cls(name, device="cpu")


def build_embeddings(
    index: Index,
    index_path: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    words: int = 200,
    overlap: int = 40,
    batch_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Embed every chunk of every document; write ``<index>.emb.*`` next to the index."""
    starts: list[tuple[int, int]] = []
    texts: list[str] = []
    for doc_id in range(index.num_docs):
        for off, chunk in chunks(index.text(doc_id) or "", words, overlap):
            starts.append((doc_id, off))
            texts.append(chunk)
    model = _model(model_name)
    out: list[npt.NDArray[np.float32]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out.append(np.asarray(vecs, dtype=np.float32))
        if progress is not None:
            progress(min(i + batch_size, len(texts)), len(texts))
    emb = np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)
    e_path, c_path, j_path = emb_paths(index_path)
    np.save(e_path, emb)
    np.save(c_path, np.asarray(starts, dtype=np.int64).reshape(-1, 2))
    j_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "chunks": len(texts),
                "dim": emb.shape[1],
                "words": words,
                "overlap": overlap,
                "docs": index.num_docs,
            }
        )
    )
    return e_path


@dataclass
class SemanticIndex:
    embeddings: npt.NDArray[np.float32]  # (n_chunks, d), rows L2-normalized
    chunk_doc: npt.NDArray[np.int64]  # (n_chunks,)
    chunk_start: npt.NDArray[np.int64]  # (n_chunks,) char offset of each chunk in its doc
    model_name: str
    _model: Any = field(default=None, repr=False)

    @classmethod
    def load(cls, e_path: Path | str) -> SemanticIndex:
        e_path = Path(e_path)
        base = Path(str(e_path).removesuffix(".emb.npy"))
        _, c_path, j_path = emb_paths(base.with_suffix(".fidx"))
        meta = json.loads(j_path.read_text())
        emb = np.load(e_path, mmap_mode="r")  # the OS pages it in as needed
        ch = np.load(c_path)
        return cls(emb, ch[:, 0], ch[:, 1], meta["model"])

    def encode(self, query: str) -> npt.NDArray[np.float32]:
        if self._model is None:
            self._model = _model(self.model_name)
        vec = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vec[0], dtype=np.float32)

    @timed
    def rank(
        self, query: str, n_docs: int = CANDIDATES
    ) -> tuple[list[tuple[int, float]], dict[int, int]]:
        """Top documents by best-chunk cosine similarity, and each doc's best chunk offset."""
        sims = self.embeddings @ self.encode(query)  # one matrix-vector product: every chunk
        n_top = min(len(sims), n_docs * 8)
        top = (
            np.argpartition(-sims, n_top - 1)[:n_top] if n_top < len(sims) else np.arange(len(sims))
        )
        top = top[np.argsort(-sims[top], kind="stable")]
        best: dict[int, float] = {}
        anchor: dict[int, int] = {}
        for c in top:  # best chunk per document (first seen = highest)
            d = int(self.chunk_doc[c])
            if d not in best:
                best[d] = float(sims[c])
                anchor[d] = int(self.chunk_start[c])
                if len(best) >= n_docs:
                    break
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0])), anchor


def rrf(*rankings: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: ``Σ 1/(k + rank)`` over the rankings a doc appears in."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def keyword_ranking(
    index: Index, query: str, scorer: Literal["bm25", "tfidf"], n: int
) -> list[int]:
    from findex.fast import search_fast

    try:
        return [r.doc_id for r in search_fast(index, query, get_scorer(scorer), n)]
    except QueryError:
        return []


@timed
def search_mode(
    index: Index,
    sem: SemanticIndex,
    query: str,
    *,
    mode: Literal["keyword", "semantic", "hybrid"],
    k: int = 10,
    offset: int = 0,
    scorer: Literal["bm25", "tfidf"] = "bm25",
) -> ResultList:
    """keyword | semantic | hybrid, all returning the same ResultList shape."""
    terms = tuple(t for t in dict.fromkeys(tokenize(query)) if t in index)
    # For highlighting only, skip words found in over a fifth of all documents ("a", "with"):
    # natural-language queries are full of them and they make snippets noisy.
    highlight = tuple(t for t in terms if index.df(t) <= max(1, index.num_docs // 5))
    if mode == "keyword":
        from findex.fast import search_fast

        return search_fast(index, query, get_scorer(scorer), k, offset=offset)
    sem_ranked, anchors = sem.rank(query, max(CANDIDATES, offset + k))
    if mode == "semantic":
        fused = sem_ranked
    else:
        # Keyword side uses OR so a paraphrase sharing *some* words still contributes.
        kw_query = " OR ".join(terms) if terms else query
        fused = rrf(
            keyword_ranking(index, kw_query, scorer, CANDIDATES), [d for d, _ in sem_ranked]
        )
    page = fused[offset : offset + k]
    hits = [SearchResult(doc_id=d, score=s, title=index.title(d)) for d, s in page]
    return ResultList(hits, len(fused), highlight, anchors)
