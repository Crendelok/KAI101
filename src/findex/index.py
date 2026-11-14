"""The inverted index (Lab 2), as a proper object (Lab 3).

Postings are stored as parallel ``array('I')`` buffers per term — doc ids, term
frequencies and (optionally) flattened token positions — not as millions of
Python objects. The Lab 2 memory study (``scripts/bench_lab02.py``) is why.
``PostingList`` still *presents* them as a ``Sequence[Posting]``.

    python -m findex.index data/sections --out index.fidx [--positions] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import zlib
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import overload, override

from findex.corpus import Document, iter_documents
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


class PostingList(Sequence[Posting]):
    """Postings for one term, sorted by doc id, stored column-wise in arrays."""

    __slots__ = ("_starts", "doc_ids", "positions", "tfs")

    def __init__(
        self,
        doc_ids: array[int] | None = None,
        tfs: array[int] | None = None,
        positions: array[int] | None = None,
    ) -> None:
        self.doc_ids: array[int] = doc_ids if doc_ids is not None else array("I")
        self.tfs: array[int] = tfs if tfs is not None else array("I")
        self.positions: array[int] | None = positions  # flattened; len == sum(tfs)
        self._starts: array[int] | None = None

    def append(self, doc_id: int, tf: int, positions: Iterable[int] | None = None) -> None:
        if self.doc_ids and doc_id <= self.doc_ids[-1]:
            raise ValueError(f"postings must be appended in doc-id order ({doc_id})")
        self.doc_ids.append(doc_id)
        self.tfs.append(tf)
        if positions is not None:
            if self.positions is None:
                self.positions = array("I")
            self.positions.extend(positions)
        self._starts = None

    def __len__(self) -> int:
        return len(self.doc_ids)

    @overload
    def __getitem__(self, i: int) -> Posting: ...
    @overload
    def __getitem__(self, i: slice) -> list[Posting]: ...
    def __getitem__(self, i: int | slice) -> Posting | list[Posting]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        return Posting(self.doc_ids[i], self.tfs[i], tuple(self.positions_at(i)))

    @override
    def __iter__(self) -> Iterator[Posting]:
        for i in range(len(self)):
            yield self[i]

    def positions_at(self, i: int) -> Sequence[int]:
        """Token positions of the i-th posting (empty if positions weren't recorded)."""
        if self.positions is None:
            return ()
        if self._starts is None:
            starts = array("I", [0])
            total = 0
            for tf in self.tfs:
                total += tf
                starts.append(total)
            self._starts = starts
        return self.positions[self._starts[i] : self._starts[i + 1]]

    def find(self, doc_id: int) -> int:
        """Index of `doc_id` in this list, or -1 (binary search)."""
        i = bisect_left(self.doc_ids, doc_id)
        return i if i < len(self.doc_ids) and self.doc_ids[i] == doc_id else -1

    @override
    def __repr__(self) -> str:
        return f"PostingList(df={len(self)}, positions={self.positions is not None})"


class Index(Mapping[str, PostingList]):
    """term -> PostingList, plus everything ranking and display need.

    ``len(ix)`` is the vocabulary size; ``"whale" in ix``; ``ix["whale"]``;
    ``for term in ix``. Doc ids are dense: ``0 .. num_docs - 1``.
    """

    def __init__(
        self,
        postings: Mapping[str, PostingList],
        doc_meta: Sequence[DocMeta],
        doc_lengths: Sequence[int],
        texts: Sequence[bytes] | None = None,
        *,
        has_positions: bool = False,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._postings = postings
        self.doc_meta = doc_meta
        self.doc_lengths = doc_lengths
        self._texts = texts
        self.has_positions = has_positions
        self._on_close = on_close

    # --- the Mapping protocol -------------------------------------------------
    @override
    def __getitem__(self, term: str) -> PostingList:
        return self._postings[term]  # KeyError if absent, like any mapping

    @override
    def __len__(self) -> int:
        return len(self._postings)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    @override
    def __contains__(self, term: object) -> bool:
        return term in self._postings

    @override
    def __repr__(self) -> str:
        return f"<Index docs={self.num_docs:,} terms={len(self):,} positions={self.has_positions}>"

    # Mapping defines __eq__ (and so sets __hash__ = None). An index is an entity,
    # not a value: identity equality keeps it hashable for caches.
    @override
    def __eq__(self, other: object) -> bool:
        return self is other

    @override
    def __hash__(self) -> int:
        return id(self)

    # --- statistics -------------------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0

    @cached_property
    def total_tokens(self) -> int:
        return sum(self.doc_lengths)

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        pl = self._postings.get(term)
        return len(pl) if pl is not None else 0

    def all_doc_ids(self) -> range:
        return range(self.num_docs)

    # --- documents ----------------------------------------------------------------
    def title(self, doc_id: int) -> str:
        return self.doc_meta[doc_id].title

    def text(self, doc_id: int) -> str | None:
        if self._texts is None:
            return None
        return zlib.decompress(self._texts[doc_id]).decode("utf-8")

    @property
    def has_texts(self) -> bool:
        return self._texts is not None

    # --- lifecycle ------------------------------------------------------------------
    def close(self) -> None:
        if self._on_close is not None:
            self._on_close()
            self._on_close = None


@timed
def build_index(
    docs: Iterable[Document], *, positions: bool = False, store_text: bool = True
) -> Index:
    """One streaming pass: the corpus flows through; only the index grows.

    Each document's counts are gathered in a per-document ``Counter`` (or a dict
    of position lists) and flushed into the postings when the document ends, so
    postings come out sorted by doc id for free.
    """
    postings: defaultdict[str, PostingList] = defaultdict(PostingList)
    doc_meta: list[DocMeta] = []
    doc_lengths = array("I")
    texts: list[bytes] = []
    for doc in docs:
        doc_id = len(doc_meta)  # dense ids, whatever the source numbered
        if positions:
            per_term: defaultdict[str, list[int]] = defaultdict(list)
            n = 0
            for n, tok in enumerate(tokenize(doc.text), start=1):
                per_term[tok].append(n - 1)
            for term, pos in per_term.items():
                postings[term].append(doc_id, len(pos), pos)
            length = n
        else:
            counts = Counter(tokenize(doc.text))
            for term, tf in counts.items():
                postings[term].append(doc_id, tf)
            length = counts.total()
        doc_meta.append(DocMeta(doc_id, doc.path, doc.title))
        doc_lengths.append(length)
        if store_text:
            texts.append(zlib.compress(doc.text.encode("utf-8"), 6))
    log.info("indexed %d documents, %d terms", len(doc_meta), len(postings))
    return Index(
        dict(postings),
        doc_meta,
        doc_lengths,
        texts if store_text else None,
        has_positions=positions,
    )


def main(argv: list[str] | None = None) -> int:
    from findex.measure import measured
    from findex.store import save

    p = argparse.ArgumentParser(prog="python -m findex.index")
    p.add_argument("corpus", type=Path)
    p.add_argument("--out", type=Path, required=True, help=".fidx (binary) or .pkl (pickle)")
    p.add_argument("--positions", action="store_true", help="record token positions")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with measured("build") as build:
        docs = iter_documents(args.corpus)
        if args.limit is not None:
            docs = islice(docs, args.limit)
        index = build_index(docs, positions=args.positions)
    with measured("save") as saving:
        save(index, args.out)
    print(index)
    print(f"build: {build}")
    print(f"save:  {saving}  ->  {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
