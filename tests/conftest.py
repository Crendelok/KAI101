from __future__ import annotations

from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index, build_index

# A tiny corpus where the right answers are obvious.
FIXTURE_DOCS = {
    "00-whale.txt": "The Whale\n\nThe white whale swam. Ahab hunted the whale across the sea.",
    "01-ship.txt": "The Ship\n\nA ship sailed the sea. The crew of the ship sang.",
    "02-cat.txt": "The Cat\n\nThe cat sat on the mat. The cat purred.",
    "03-long.txt": "A Long One\n\n" + "filler words go here " * 60 + "whale",
    "04-rare.txt": "Rare\n\nNarwhal sightings near the sea are rare.",
    "05-repeat.txt": "Repeat\n\n" + "whale " * 20 + "ocean",
    "06-phrase.txt": "Phrase\n\nThe sea was white, and the white whale was not.",
    "07-ukr.txt": "Кит\n\nБілий кит пливе у морі. П'ять китів.",
}


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for name, text in FIXTURE_DOCS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def index(corpus_dir: Path) -> Index:
    """Session-scoped and therefore shared: tests must never mutate it."""
    return build_index(iter_documents(corpus_dir), positions=True)


@pytest.fixture(scope="session")
def index_nopos(corpus_dir: Path) -> Index:
    return build_index(iter_documents(corpus_dir), positions=False)


def doc_named(index: Index, stem: str) -> int:
    for meta in index.doc_meta:
        if Path(meta.path).stem == stem:
            return meta.doc_id
    raise KeyError(stem)
