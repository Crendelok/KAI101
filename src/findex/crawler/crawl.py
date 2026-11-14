"""The crawl: an ``asyncio.Queue`` frontier drained by worker tasks in a ``TaskGroup``.

Politeness, in layers:

* a **global semaphore** caps requests in flight;
* a **per-host semaphore** caps requests in flight to any one ``host:port``;
* a **per-host minimum delay** spaces out request *starts* to a host
  (raised to ``Crawl-delay`` if robots.txt asks for more);
* ``robots.txt`` is obeyed, only allowed domains are followed, and the
  budget (``max_pages`` fetches) is never exceeded.

``crawl()`` is an async generator: pages are yielded as they arrive, so the
caller can stream them to disk without ever holding the crawl in memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.html import Extracted, extract
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, domain_of, host_of, seen_key

log = logging.getLogger(__name__)
url_log = logging.getLogger("findex.crawl.urls")  # one line per URL -> crawl.log

DEFAULT_CONTACT = "https://github.com/OWNER/REPO"


def user_agent(contact: str = DEFAULT_CONTACT) -> str:
    from findex import __version__

    return f"findex-crawler/{__version__} (+{contact}; polite educational crawler)"


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str
    status: int


@dataclass
class CrawlStats:
    fetched: int = 0
    pages: int = 0
    errors: int = 0
    retried: int = 0
    robots_blocked: int = 0
    in_flight: int = 0
    queued: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def pages_per_second(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0


class _HostPacer:
    """Per-host semaphore + minimum spacing between request starts."""

    def __init__(self, per_host: int, min_delay: float) -> None:
        self.per_host = per_host
        self.min_delay = min_delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._next: dict[str, float] = {}

    def semaphore(self, host: str) -> asyncio.Semaphore:
        return self._sems.setdefault(host, asyncio.Semaphore(self.per_host))

    async def wait_turn(self, host: str, delay: float) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        slot = max(now, self._next.get(host, now))  # no await between read and write:
        self._next[host] = slot + delay  # atomic under the event loop, no lock needed
        if slot > now:
            await asyncio.sleep(slot - now)


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    min_delay: float = 0.0,
    allowed_domains: Iterable[str] | None = None,
    client: httpx.AsyncClient | None = None,
    contact: str = DEFAULT_CONTACT,
    stats: CrawlStats | None = None,
    timeout: float = 10.0,
) -> AsyncIterator[Page]:
    """Breadth-first crawl from `seeds`; yields each successfully fetched HTML page."""
    agent = user_agent(contact)
    own_client = client is None
    # httpx imports its HTTP/1.1 + anyio machinery lazily on the first request — ~0.5 s of
    # cold imports that would otherwise run *on the event loop*. Pay it in a thread instead.
    await asyncio.to_thread(_warm_imports)
    if client is None:
        # Building a client loads the TLS trust store (~0.3 s of blocking file I/O and
        # parsing): do it in a thread so the event loop is never stalled, even at startup.
        client = await asyncio.to_thread(
            httpx.AsyncClient,
            headers={"User-Agent": agent},
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            timeout=timeout,
        )
    stats = stats if stats is not None else CrawlStats()
    seed_urls = [u for u in (canonicalize(s) for s in seeds) if u]
    domains = (
        {d.lower() for d in allowed_domains} if allowed_domains else set(map(domain_of, seed_urls))
    )
    robots = RobotsCache(client, agent)
    pacer = _HostPacer(per_host, min_delay)
    global_sem = asyncio.Semaphore(concurrency)
    frontier: asyncio.Queue[str] = asyncio.Queue()
    results: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    emitted: set[str] = set()  # final URLs already yielded (after redirects)
    budget_left = max_pages
    claimed_active = 0  # budget claimed by visits that haven't finished yet
    budget_spent = asyncio.Event()

    def enqueue(url: str) -> None:
        key = seen_key(url)
        if key in seen or domain_of(url) not in domains:
            return
        seen.add(key)
        frontier.put_nowait(url)
        stats.queued = frontier.qsize()

    for url in seed_urls:
        enqueue(url)

    async def visit(url: str) -> None:
        nonlocal budget_left, claimed_active
        if not await robots.allowed(url):
            stats.robots_blocked += 1
            url_log.info("ROBOTS %s", url)
            return
        if budget_left <= 0:
            return
        budget_left -= 1  # claim before awaiting: the budget can't be overshot
        claimed_active += 1
        try:
            await _fetch_and_parse(url)
        finally:
            claimed_active -= 1
            if budget_left <= 0:
                budget_spent.set()

    async def _fetch_and_parse(url: str) -> None:
        host = host_of(url)
        delay = max(min_delay, await robots.crawl_delay(url) or 0.0)
        async with global_sem, pacer.semaphore(host):
            await pacer.wait_turn(host, delay)
            stats.in_flight += 1
            try:
                res = await fetch(client, url, timeout=timeout)
            finally:
                stats.in_flight -= 1
        stats.fetched += 1
        if res.attempts > 1:
            stats.retried += 1
        _log_result(res)
        if not res.ok:
            stats.errors += 1
            return
        if not res.is_html:
            return
        final_key = seen_key(res.final_url)
        if final_key != seen_key(url):  # redirected: don't emit (or later fetch) the target twice
            if final_key in emitted:
                return
            seen.add(final_key)
        emitted.add(final_key)
        # Parsing is CPU work: keep it off the event loop's thread.
        doc: Extracted = await asyncio.to_thread(extract, res.body.decode("utf-8", "replace"))
        for href in doc.links:
            if (link := canonicalize(href, res.final_url)) is not None:
                enqueue(link)
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        stats.pages += 1
        await results.put(Page(res.final_url, doc.title or url, doc.text, stamp, res.status))

    async def worker() -> None:
        skipped = 0
        while True:
            url = await frontier.get()
            # Head-of-line blocking fix: if this URL's host is already at its per-host
            # limit and other URLs are waiting, put it back and take another — otherwise
            # every worker ends up parked on the busiest host's semaphore.
            if pacer.semaphore(host_of(url)).locked() and skipped < frontier.qsize():
                frontier.put_nowait(url)
                frontier.task_done()
                skipped += 1
                await asyncio.sleep(0)
                continue
            if skipped:  # we went all the way round: every queued host is busy, so wait
                await asyncio.sleep(0.005)
            skipped = 0
            try:
                await visit(url)
            # One bad page must not kill the crawl; CancelledError is a BaseException
            # and still propagates, so shutdown keeps working.
            except Exception:
                stats.errors += 1
                log.exception("crawling %s failed", url)
            finally:
                frontier.task_done()
                stats.queued = frontier.qsize()

    async def run() -> None:
        async with asyncio.TaskGroup() as tg:
            workers = [tg.create_task(worker()) for _ in range(concurrency)]
            drained = tg.create_task(frontier.join())
            spent = tg.create_task(budget_spent.wait())
            await asyncio.wait({drained, spent}, return_when=asyncio.FIRST_COMPLETED)
            if budget_spent.is_set():
                # let visits that already claimed budget finish, then stop
                while claimed_active:
                    await asyncio.sleep(0.01)
            for t in (*workers, drained, spent):
                t.cancel()
        await results.put(None)

    runner = asyncio.create_task(run())
    try:
        while (page := await results.get()) is not None:
            yield page
        await runner  # surface any exception from the TaskGroup
    finally:
        if not runner.done():
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner
        if own_client:
            await client.aclose()


def _warm_imports() -> None:
    import importlib

    for mod in ("h11", "httpcore._async.http11", "httpcore._backends.anyio", "html.parser"):
        # The transport's module layout differs between versions: best effort.
        with contextlib.suppress(ImportError):
            importlib.import_module(mod)


def _log_result(res: FetchResult) -> None:
    status = res.status if res.status else (res.error or "ERR")
    retry = f" attempts={res.attempts}" if res.attempts > 1 else ""
    url_log.info(
        "%s %s %.0fms%s%s",
        status,
        res.url,
        res.elapsed * 1000,
        retry,
        f" -> {res.final_url}" if res.final_url != res.url else "",
    )
