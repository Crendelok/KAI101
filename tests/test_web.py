"""API + UI tests. The fixture index is injected with dependency_overrides — no index file."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web.app import create_app, get_index_optional
from findex.web.settings import Settings

from .conftest import doc_named


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    return create_app(Settings(index_path=tmp_path / "missing.fidx"))


@pytest.fixture
def client(app: FastAPI, index: Index) -> Iterator[TestClient]:
    app.dependency_overrides[get_index_optional] = lambda: index
    with TestClient(app) as c:
        yield c


def test_search(client: TestClient, index: Index):
    r = client.get("/search", params={"q": "white whale", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2 and body["pages"] == 1 and len(body["results"]) == 2
    top = body["results"][0]
    assert top["doc_id"] in {doc_named(index, "00-whale"), doc_named(index, "06-phrase")}
    assert "<mark>" in top["snippet_html"] and "x-request-id" in r.headers


def test_search_pagination(client: TestClient):
    all_ids = [h["doc_id"] for h in client.get("/search?q=the&k=50").json()["results"]]
    p1 = client.get("/search?q=the&k=3&page=1").json()
    p2 = client.get("/search?q=the&k=3&page=2").json()
    assert [h["doc_id"] for h in p1["results"] + p2["results"]] == all_ids[:6]
    assert p1["pages"] == -(-p1["total"] // 3)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "x", "k": 0},
        {"q": "x", "k": 51},
        {"q": "x", "page": 0},
        {"q": "x", "scorer": "magic"},
        {"q": "x", "mode": "psychic"},
        {"q": "x" * 501},
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, object]):
    r = client.get("/search", params=params)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_bad_query_400_and_semantic_disabled(client: TestClient):
    r = client.get("/search", params={"q": "(whale"})
    assert r.status_code == 400 and "bad query" in r.json()["detail"]
    r = client.get("/search", params={"q": "whale", "mode": "semantic"})
    assert r.status_code == 400 and "not enabled" in r.json()["detail"]


def test_document_and_404(client: TestClient, index: Index):
    d = doc_named(index, "02-cat")
    r = client.get(f"/docs/{d}")
    assert r.status_code == 200 and r.json()["title"] == "The Cat"
    assert client.get("/docs/9999").status_code == 404
    assert client.get("/docs/not-a-number").status_code == 422


def test_stats(client: TestClient):
    body = client.get("/stats").json()
    assert body["documents"] == 8 and body["positions"] is True and body["semantic"] is False


def test_health_ok_and_503_when_unloaded(client: TestClient, app: FastAPI):
    assert client.get("/health").json() == {"status": "ok"}
    app.dependency_overrides.clear()  # back to the real (unloaded) index
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["detail"] == "index not loaded"
    assert client.get("/search?q=whale").status_code == 503


def test_unexpected_error_is_clean_500(client: TestClient, app: FastAPI):
    def broken() -> Index:
        raise RuntimeError("boom")

    app.dependency_overrides[get_index_optional] = broken
    r = client.get("/stats", headers={"x-request-id": "abc123"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error", "request_id": "abc123"}


def test_html_ui(client: TestClient, index: Index):
    home = client.get("/")
    assert home.status_code == 200 and "Search 8 documents" in home.text
    page = client.get("/", params={"q": "whale"})
    assert "<mark>whale</mark>" in page.text.lower() or "<mark>Whale</mark>" in page.text
    doc = client.get(f"/doc/{doc_named(index, '00-whale')}", params={"q": "ahab"})
    assert doc.status_code == 200 and "<mark>Ahab</mark>" in doc.text
    assert client.get("/static/style.css").status_code == 200


def test_openapi_docs(client: TestClient):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= spec["paths"].keys()


def test_async_variant_gives_same_answers(tmp_path: Path, index: Index):
    app = create_app(Settings(index_path=tmp_path / "x.fidx", search_async=True))
    app.dependency_overrides[get_index_optional] = lambda: index
    with TestClient(app) as c:
        assert c.get("/search?q=whale").json()["total"] == 4
