"""A lazy document stream (Lab 1).

`iter_documents(root)` is a generator: it opens one file at a time, yields its
document(s), and moves on. Nothing about the corpus is held in memory except
the document currently being yielded.

Supported inputs (a single file, or any mix of these found under a directory):

* ``*.txt``                      — one file = one document; first non-empty line = title
* ``*.jsonl``                    — one line = one document (``{"title": ..., "text": ...}``)
* ``*.txt.gz`` / ``*.txt.bz2``   — compressed, read without extracting
* ``*.jsonl.gz`` / ``*.jsonl.bz2``

Undecodable bytes never crash the stream: the file (or line) is logged and
re-decoded with ``errors="replace"``; malformed JSON lines are logged and skipped.

Files are visited in sorted path order so doc ids are deterministic — which is
what lets Lab 5's parallel build produce an index identical to the serial one.
"""

from __future__ import annotations

import bz2
import gzip
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, NamedTuple

log = logging.getLogger(__name__)

TEXT_KEYS = ("text", "abstract", "body", "content")
TITLE_MAX = 120


class Document(NamedTuple):
    doc_id: int
    path: str
    title: str
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


def _open_binary(path: Path, compression: str) -> IO[bytes] | gzip.GzipFile | bz2.BZ2File:
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


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line := line.strip():
            return line[:TITLE_MAX]
    return ""


def _iter_txt(path: Path, compression: str) -> Iterator[tuple[str, str, str]]:
    with _open_binary(path, compression) as f:
        text = _decode(f.read(), str(path))
    yield str(path), first_line(text) or path.name, text


def _iter_jsonl(path: Path, compression: str) -> Iterator[tuple[str, str, str]]:
    with _open_binary(path, compression) as f:
        for lineno, raw in enumerate(f, start=1):  # a file is a lazy iterator over lines
            if not raw.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                record: object = json.loads(_decode(raw, where))
            except json.JSONDecodeError as exc:
                log.warning("skipping malformed JSON at %s: %s", where, exc.msg)
                continue
            text: str | None = None
            title = ""
            if isinstance(record, str):
                text = record
            elif isinstance(record, dict):
                for key in TEXT_KEYS:
                    value = record.get(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    if isinstance(value, str):
                        text = value
                        break
                t = record.get("title")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                title = t[:TITLE_MAX] if isinstance(t, str) else ""
                url = record.get("url")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if isinstance(url, str):
                    where = url
            if text is None:
                log.warning("skipping %s: no text field (%s)", where, ", ".join(TEXT_KEYS))
                continue
            yield where, title or first_line(text) or where, text


def iter_paths(root: Path | str) -> list[Path]:
    """All supported corpus files under `root`, in sorted order (paths only — cheap)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return [root] if _kind(root) else []
    return sorted(p for p in root.rglob("*") if p.is_file() and _kind(p))


def iter_documents_from(paths: Iterable[Path], start_id: int = 0) -> Iterator[Document]:
    """Yield the documents in `paths`, numbering them from `start_id`."""
    doc_id = start_id
    for path in paths:
        kind = _kind(path)
        if kind is None:
            continue
        fmt, compression = kind
        reader = _iter_jsonl if fmt == ".jsonl" else _iter_txt
        try:
            for where, title, text in reader(path, compression):
                yield Document(doc_id, where, title, text)
                doc_id += 1
        except OSError as exc:  # unreadable / truncated archive: log and keep going
            log.error("cannot read %s: %s", path, exc)


def iter_documents(root: Path | str) -> Iterator[Document]:
    """Yield every document under `root`, one at a time."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        yield from iter_documents_from([root])
    else:
        # Sorting needs the list of *paths* (bytes per file), never the documents.
        yield from iter_documents_from(iter_paths(root))
