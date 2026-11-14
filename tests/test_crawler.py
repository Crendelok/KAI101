"""Crawler tests: no network — every response comes from httpx.MockTransport."""

from __future__ import annotations

import asyncio
import importlib
from collections import Counter
from collections.abc import Callable

import httpx
import pytest

from findex.crawler.crawl import CrawlStats, Page, crawl
from findex.crawler.fetch import fetch, retry_after_seconds
from findex.crawler.html import extract
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, seen_key

# findex.crawler re-exports the *functions* `fetch` and `crawl`, which shadow the
# submodules of the same name as package attributes — so get the modules explicitly.
fetch_mod = importlib.import_module("findex.crawler.fetch")
crawl_mod = importlib.import_module("findex.crawler.crawl")

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of waiting for them."""
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float, *args: object) -> None:
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(fetch_mod.asyncio, "sleep", fake_sleep)
    return delays


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run_fetch(handler: Handler, url: str = "http://x.test/a", **kw: object):
    async def go():
        async with client_for(handler) as c:
            return await fetch(c, url, **kw)  # type: ignore[arg-type]

    return asyncio.run(go())


# --- urls --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "base", "expected"),
    [
        ("HTTP://Example.COM", None, "http://example.com/"),
        ("http://example.com:80/a/./b/../c#frag", None, "http://example.com/a/c"),
        ("https://e.com:443/x?b=2&a=1", None, "https://e.com/x?a=1&b=2"),
        ("../up", "http://e.com/a/b/c", "http://e.com/a/up"),
        ("2", "http://e.com/book/1/", "http://e.com/book/1/2"),
        ("mailto:a@b.c", None, None),
        ("javascript:void(0)", None, None),
        ("http://e.com:8101/book/", None, "http://e.com:8101/book/"),
    ],
)
def test_canonicalize(raw: str, base: str | None, expected: str | None):
    assert canonicalize(raw, base) == expected


def test_seen_key_ignores_trailing_slash():
    assert seen_key("http://e.com/a/") == seen_key("http://e.com/a")
    assert seen_key("http://e.com/") == "http://e.com/"


# --- fetch ----------------------------------------------------------------------------


def test_fetch_ok():
    res = run_fetch(
        lambda r: httpx.Response(200, text="<p>hi</p>", headers={"content-type": "text/html"})
    )
    assert res.ok and res.is_html and res.attempts == 1 and res.body == b"<p>hi</p>"


def test_retry_after_is_honored(no_real_sleep: list[float]):
    calls = Counter[str]()

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, text="ok")

    res = run_fetch(handler)
    assert res.ok and res.attempts == 2
    assert no_real_sleep == [7.0]


def test_5xx_retries_with_growing_backoff(no_real_sleep: list[float]):
    res = run_fetch(lambda r: httpx.Response(503), max_retries=3, base_delay=1.0)
    assert res.status == 503 and res.attempts == 4 and not res.ok
    assert len(no_real_sleep) == 3
    assert 1 <= no_real_sleep[0] < 2 <= no_real_sleep[1] < 3 and 4 <= no_real_sleep[2] < 5


def test_404_is_not_retried(no_real_sleep: list[float]):
    res = run_fetch(lambda r: httpx.Response(404))
    assert res.status == 404 and res.attempts == 1 and no_real_sleep == []


def test_timeouts_retry_then_give_up(no_real_sleep: list[float]):
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=req)

    res = run_fetch(handler, max_retries=2)
    assert res.status == 0 and res.attempts == 3 and res.error == "ConnectTimeout"


def test_redirect_loop_is_an_error_not_a_retry():
    res = run_fetch(lambda r: httpx.Response(301, headers={"Location": str(r.url)}))
    assert res.status == 0 and res.error and "TooManyRedirects" in res.error


def test_retry_after_http_date():
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0
    assert retry_after_seconds("120") == 60.0  # capped
    assert retry_after_seconds("soon") is None


# --- robots + html -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "allowed_private", "allowed_public"),
    [
        (200, "User-agent: *\nDisallow: /private/", False, True),
        (404, "", True, True),
        (500, "", False, False),
    ],
)
def test_robots(status: int, body: str, allowed_private: bool, allowed_public: bool):
    async def go():
        async with client_for(lambda r: httpx.Response(status, text=body)) as c:
            robots = RobotsCache(c, "findex-test")
            return (
                await robots.allowed("http://x.test/private/a"),
                await robots.allowed("http://x.test/public"),
            )

    assert asyncio.run(go()) == (allowed_private, allowed_public)


def test_extract():
    doc = extract(
        "<html><head><title> T </title><script>x()</script></head><body>"
        "<nav><a href='/n'>nav</a></nav><p>Hello <b>world</b></p><p>Two</p>"
        "<a href='/x'>link</a></body></html>"
    )
    assert doc.title == "T"
    assert doc.text == "Hello world\n\nTwo\n\nlink"
    assert doc.links == ["/n", "/x"]


# --- the crawl over a fake site ------------------------------------------------------------


def fake_site(hits: Counter[str]) -> Handler:
    pages = {
        "/": "<a href='/a'>a</a> <a href='/b#x'>b</a> <a href='/private/s'>s</a>"
        "<a href='http://other.test/'>other</a> <a href='/a?'>a again</a>",
        "/a": "<title>A</title><a href='/b'>b</a><a href='/c'>c</a><a href='/'>home</a>",
        "/b": "<title>B</title><a href='/c/'>c</a>",
        "/c": "<title>C</title>text c",
        "/private/s": "secret",
    }

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        hits[f"{req.url.host}{path}"] += 1
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/")
        body = pages.get(path.rstrip("/") or "/")
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    return handler


def run_crawl(handler: Handler, **kw: object) -> tuple[list[Page], CrawlStats]:
    stats = CrawlStats()

    async def go() -> list[Page]:
        async with client_for(handler) as c:
            return [p async for p in crawl(["http://site.test/"], client=c, stats=stats, **kw)]  # type: ignore[arg-type]

    return asyncio.run(go()), stats


def test_crawl_visits_each_page_once_and_obeys_robots_and_domains():
    hits = Counter[str]()
    pages, stats = run_crawl(fake_site(hits), concurrency=3, per_host=2)
    assert sorted(p.title for p in pages) == ["A", "B", "C", "http://site.test/"]
    assert all(n == 1 for n in hits.values()), hits
    assert "site.test/private/s" not in hits
    assert not any(h.startswith("other.test") for h in hits)
    assert stats.robots_blocked == 1 and stats.errors == 0


def test_crawl_stops_at_budget():
    hits = Counter[str]()
    pages, stats = run_crawl(fake_site(hits), max_pages=2, concurrency=4)
    assert len(pages) == 2 and stats.fetched == 2
    assert sum(n for k, n in hits.items() if not k.endswith("robots.txt")) == 2


def test_per_host_limit_is_respected(monkeypatch: pytest.MonkeyPatch):
    in_flight = peak = 0

    async def slow_fetch(client: httpx.AsyncClient, url: str, **kw: object):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        links = "".join(f"<a href='/p{i}'>x</a>" for i in range(20))
        return fetch_mod.FetchResult(url, url, 200, "text/html", links.encode(), 0.01, 1)

    monkeypatch.setattr(crawl_mod, "fetch", slow_fetch)
    pages, _ = run_crawl(lambda r: httpx.Response(404), concurrency=10, per_host=3, max_pages=15)
    assert len(pages) == 15
    assert peak == 3
