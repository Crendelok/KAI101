from __future__ import annotations

from pathlib import Path

import pytest

from findex import parallel
from findex.corpus import iter_documents
from findex.index import build_index
from findex.parallel import build_parallel, chunked
from findex.store import save


def fidx_bytes(index: object, path: Path) -> bytes:
    save(index, path)  # type: ignore[arg-type]
    return path.read_bytes()


@pytest.fixture(scope="module")
def serial_bytes(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Lab 4's single-pass build is the reference."""
    ix = build_index(iter_documents(corpus_dir), positions=True)
    return fidx_bytes(ix, tmp_path_factory.mktemp("ref") / "serial.fidx")


def test_chunked_preserves_order_and_covers_everything():
    items = [Path(str(i)) for i in range(10)]
    chunks = chunked(items, 3)
    assert [len(c) for c in chunks] == [4, 3, 3]
    assert [p for c in chunks for p in c] == items
    assert chunked(items, 50) == [[p] for p in items]


@pytest.mark.parametrize(("executor", "workers"), [("serial", 1), ("threads", 3)])
def test_parallel_build_is_byte_identical(
    corpus_dir: Path, tmp_path: Path, serial_bytes: bytes, executor: str, workers: int
):
    ix, report = build_parallel(corpus_dir, workers=workers, executor=executor, chunks_per_worker=2)  # type: ignore[arg-type]
    assert fidx_bytes(ix, tmp_path / "p.fidx") == serial_bytes
    assert report.chunks == (1 if executor == "serial" else 6)


@pytest.mark.slow
def test_process_build_is_byte_identical(corpus_dir: Path, tmp_path: Path, serial_bytes: bytes):
    ix, report = build_parallel(corpus_dir, workers=2, executor="processes", chunks_per_worker=2)
    assert fidx_bytes(ix, tmp_path / "p.fidx") == serial_bytes
    assert report.cpu > 0


def test_worker_exception_surfaces(corpus_dir: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*args: object) -> object:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(parallel, "build_partial", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        build_parallel(corpus_dir, workers=2, executor="threads")


def test_progress_callback_counts_documents(corpus_dir: Path):
    seen: list[int] = []
    build_parallel(corpus_dir, workers=2, executor="threads", on_chunk=seen.append)
    assert sum(seen) == 8
