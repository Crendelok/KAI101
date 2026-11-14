"""M1 — a lazy document stream.

`iter_documents(root)` is a generator: it opens one file at a time, yields its
document(s), and moves on. Nothing about the corpus is held in memory except
the document currently being yielded.

Supported inputs (a single file, or any mix of these found under a directory):

* ``*.txt``                      — one file = one document
* ``*.jsonl``                    — one line = one document (``{"id": ..., "text": ...}``)
* ``*.txt.gz`` / ``*.txt.bz2``   — compressed, read without extracting (stretch)
* ``*.jsonl.gz`` / ``*.jsonl.bz2``

Undecodable bytes never crash the stream: the file (or line) is logged and
re-decoded with ``errors="replace"``; malformed JSON lines are logged and skipped.
"""

from __future__ import annotations

import bz2
import gzip
import json
import logging
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import IO, NamedTuple

log = logging.getLogger(__name__)

# Keys tried, in order, to find the text of a JSON-lines record.
TEXT_KEYS = ("text", "abstract", "body", "content")


class Document(NamedTuple):
    doc_id: str
    path: str
    text: str


def _kind(path: Path) -> tuple[str, str] | None:
    """Return (format, compression) for a supported file, else None."""
    suffixes = [s.lower() for s in path.suffixes[-2:]]
    compression = ""
    if suffixes and suffixes[-1] in (".gz", ".bz2"):
        compression = suffixes.pop()
    if suffixes and suffixes[-1] in (".txt", ".jsonl"):
        return suffixes[-1], compression
    return None


def _open_binary(path: Path, compression: str) -> IO[bytes]:
    if compression == ".gz":
        return gzip.open(path, "rb")
    if compression == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def _decode(raw: bytes, where: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        log.warning("bad UTF-8 in %s (%s); replacing undecodable bytes", where, exc.reason)
        return raw.decode("utf-8", errors="replace")


def _iter_txt(path: Path, compression: str) -> Iterator[tuple[str, str]]:
    with _open_binary(path, compression) as f:
        yield str(path), _decode(f.read(), str(path))


def _iter_jsonl(path: Path, compression: str) -> Iterator[tuple[str, str]]:
    with _open_binary(path, compression) as f:
        for lineno, raw in enumerate(f, start=1):  # a file is a lazy iterator over lines
            if not raw.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                record = json.loads(_decode(raw, where))
            except json.JSONDecodeError as exc:
                log.warning("skipping malformed JSON at %s: %s", where, exc.msg)
                continue
            if isinstance(record, str):
                text = record
            elif isinstance(record, dict):
                text = next((record[k] for k in TEXT_KEYS if isinstance(record.get(k), str)), None)
            else:
                text = None
            if text is None:
                log.warning("skipping %s: no text field (%s)", where, ", ".join(TEXT_KEYS))
                continue
            yield where, text


def _iter_paths(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
    else:
        # rglob is itself a lazy generator over the directory tree.
        yield from (p for p in root.rglob("*") if p.is_file())


def iter_documents(root: Path | str) -> Iterator[Document]:
    """Yield every document under `root`, one at a time."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    ids = count()
    for path in _iter_paths(root):
        kind = _kind(path)
        if kind is None:
            continue
        fmt, compression = kind
        reader = _iter_jsonl if fmt == ".jsonl" else _iter_txt
        try:
            for where, text in reader(path, compression):
                yield Document(doc_id=str(next(ids)), path=where, text=text)
        except OSError as exc:  # unreadable / truncated archive: log and keep going
            log.error("cannot read %s: %s", path, exc)
