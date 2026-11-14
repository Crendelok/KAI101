"""Persisting the index (Lab 2): pickle, and a binary ``.fidx`` format.

pickle
    One call each way, and **never safe on untrusted input**: unpickling can
    call any importable callable with any arguments (an object's
    ``__reduce__`` says "call os.system('…') to rebuild me"). Loading a
    downloaded ``.pkl`` is remote code execution. Only load pickles you wrote.

.fidx (binary, pure data — safe to load from anywhere)
    ``MAGIC | body | header JSON | u64 header length | MAGIC``

    *body*   — per term: ``doc_ids`` then ``tfs`` (uint32 each), then flattened
               positions; ``doc_lengths`` (uint32); text offsets (uint64) and
               zlib-compressed document texts.
    *header* — counts, doc metadata, and ``term -> [offset, df, pos_offset, n_pos]``.

    Loading can be eager (copy every posting list into memory) or
    ``lazy=True``: ``mmap`` the file and slice a term's postings only when it
    is looked up, so load time depends on the vocabulary, not on index size.
"""

from __future__ import annotations

import json
import mmap
import pickle
import struct
import sys
from array import array
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast, overload, override

from findex.index import DocMeta, Index, PostingList
from findex.timing import timed

MAGIC = b"FIDX\x01\n"
U64 = struct.Struct("<Q")
type Format = Literal["fidx", "pickle"]


def format_for(path: Path) -> Format:
    return "pickle" if path.suffix in (".pkl", ".pickle") else "fidx"


# --- pickle -----------------------------------------------------------------------


def save_pickle(index: Index, path: Path) -> None:
    texts = [index._texts[i] for i in range(index.num_docs)] if index._texts else None  # pyright: ignore[reportPrivateUsage]
    state = (
        dict(index.items()),
        list(index.doc_meta),
        array("I", index.doc_lengths),
        texts,
        index.has_positions,
    )
    with path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    # SECURITY: pickle.load executes code chosen by whoever wrote the file.
    # Only ever load pickles this program produced on this machine.
    with path.open("rb") as f:
        postings, doc_meta, doc_lengths, texts, has_positions = pickle.load(f)  # noqa: S301
    return Index(postings, doc_meta, doc_lengths, texts, has_positions=has_positions)


# --- .fidx ------------------------------------------------------------------------


def _as_native(buf: memoryview | bytes, typecode: str) -> array[int]:
    a = array(typecode)
    a.frombytes(buf)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def _le_bytes(a: array[int]) -> bytes:
    if sys.byteorder == "little":
        return a.tobytes()
    b = array(a.typecode, a)
    b.byteswap()
    return b.tobytes()


def save_fidx(index: Index, path: Path) -> None:
    terms: dict[str, list[int]] = {}
    with path.open("wb") as f:
        f.write(MAGIC)
        base = f.tell()
        for term in sorted(index):
            pl = index[term]
            off = f.tell() - base
            f.write(_le_bytes(pl.doc_ids))
            f.write(_le_bytes(pl.tfs))
            pos_off = f.tell() - base
            n_pos = 0
            if pl.positions is not None:
                f.write(_le_bytes(pl.positions))
                n_pos = len(pl.positions)
            terms[term] = [off, len(pl), pos_off, n_pos]
        lengths_off = f.tell() - base
        f.write(_le_bytes(array("I", index.doc_lengths)))
        texts_off = -1
        if index.has_texts:
            blobs = index._texts  # pyright: ignore[reportPrivateUsage]
            assert blobs is not None
            offsets = array("Q", [0])
            for i in range(index.num_docs):
                offsets.append(offsets[-1] + len(blobs[i]))
            texts_off = f.tell() - base
            f.write(_le_bytes(offsets))
            for i in range(index.num_docs):
                f.write(blobs[i])
        header = {
            "version": 1,
            "num_docs": index.num_docs,
            "has_positions": index.has_positions,
            "doc_meta": [[m.path, m.title] for m in index.doc_meta],
            "doc_lengths": lengths_off,
            "texts": texts_off,
            "terms": terms,
        }
        raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        f.write(raw)
        f.write(U64.pack(len(raw)))
        f.write(MAGIC)


class _LazyPostings(Mapping[str, PostingList]):
    """Postings sliced out of an mmap on first lookup, then cached."""

    def __init__(self, body: memoryview, terms: dict[str, list[int]]) -> None:
        self._body = body
        self._terms = terms
        self._cache: dict[str, PostingList] = {}

    @override
    def __getitem__(self, term: str) -> PostingList:
        pl = self._cache.get(term)
        if pl is None:
            pl = _read_postings(self._body, self._terms[term])
            self._cache[term] = pl
        return pl

    @override
    def __len__(self) -> int:
        return len(self._terms)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._terms)

    @override
    def __contains__(self, term: object) -> bool:
        return term in self._terms


def _read_postings(body: memoryview, entry: list[int]) -> PostingList:
    off, df, pos_off, n_pos = entry
    doc_ids = _as_native(body[off : off + 4 * df], "I")
    tfs = _as_native(body[off + 4 * df : off + 8 * df], "I")
    positions = _as_native(body[pos_off : pos_off + 4 * n_pos], "I") if n_pos else None
    return PostingList(doc_ids, tfs, positions)


class _Texts(Sequence[bytes]):
    def __init__(self, body: memoryview, offset: int, n: int) -> None:
        self._offsets = _as_native(body[offset : offset + 8 * (n + 1)], "Q")
        self._data = body[offset + 8 * (n + 1) :]
        self._n = n

    @override
    def __len__(self) -> int:
        return self._n

    @overload
    def __getitem__(self, i: int) -> bytes: ...
    @overload
    def __getitem__(self, i: slice) -> Sequence[bytes]: ...
    @override
    def __getitem__(self, i: int | slice) -> bytes | Sequence[bytes]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        return bytes(self._data[self._offsets[i] : self._offsets[i + 1]])

    def release(self) -> None:
        self._data.release()


def load_fidx(path: Path, *, lazy: bool = False) -> Index:
    with path.open("rb") as f:
        if lazy:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            buf = memoryview(mm)
        else:
            mm = None
            buf = memoryview(f.read())
    tail = len(MAGIC) + U64.size
    if buf[: len(MAGIC)] != MAGIC or buf[-len(MAGIC) :] != MAGIC:
        raise ValueError(f"{path} is not a findex .fidx file")
    (hlen,) = U64.unpack(buf[-tail : -len(MAGIC)])
    header = cast(dict[str, Any], json.loads(bytes(buf[-tail - hlen : -tail])))
    body = buf[len(MAGIC) : -tail - hlen]
    n = int(header["num_docs"])
    terms = cast(dict[str, list[int]], header["terms"])
    doc_meta = [DocMeta(i, p, t) for i, (p, t) in enumerate(header["doc_meta"])]
    lengths_off = int(header["doc_lengths"])
    doc_lengths = _as_native(body[lengths_off : lengths_off + 4 * n], "I")
    texts_off = int(header["texts"])
    texts = _Texts(body, texts_off, n) if texts_off >= 0 else None
    if lazy:
        postings: Mapping[str, PostingList] = _LazyPostings(body, terms)
    else:
        postings = {t: _read_postings(body, e) for t, e in terms.items()}
        if texts is not None:
            texts = [texts[i] for i in range(n)]  # own the bytes; drop the buffer

    def close() -> None:
        if mm is not None:
            if isinstance(texts, _Texts):
                texts.release()
            body.release()
            buf.release()
            mm.close()

    return Index(
        postings,
        doc_meta,
        doc_lengths,
        texts,
        has_positions=bool(header["has_positions"]),
        on_close=close if lazy else None,
    )


# --- front door -------------------------------------------------------------------


@timed
def save(index: Index, path: Path | str, fmt: Format | None = None) -> None:
    path = Path(path)
    if (fmt or format_for(path)) == "pickle":
        save_pickle(index, path)
    else:
        save_fidx(index, path)


@timed
def load(path: Path | str, fmt: Format | None = None, *, lazy: bool = False) -> Index:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if (fmt or format_for(path)) == "pickle":
        return load_pickle(path)
    return load_fidx(path, lazy=lazy)


@contextmanager
def open_index(path: Path | str, *, lazy: bool = True) -> Iterator[Index]:
    """``with open_index("index.fidx") as ix: ...`` — the mmap is closed on exit,
    including when the body raises."""
    index = load(path, lazy=lazy)
    try:
        yield index
    finally:
        index.close()
