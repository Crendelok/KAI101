"""One polite fetch: timeout, bounded retries with backoff + jitter, ``Retry-After``."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 60.0


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int  # 0 = no HTTP response (timeout, connection error, redirect loop)
    content_type: str
    body: bytes
    elapsed: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type


def retry_after_seconds(value: str | None) -> float | None:
    """``Retry-After`` is either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return min(float(value), MAX_RETRY_AFTER)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, min((when - datetime.now(UTC)).total_seconds(), MAX_RETRY_AFTER))


def backoff(attempt: int, base: float) -> float:
    """Exponential backoff with full jitter: ``base * 2**(n-1)`` plus up to ``base`` extra."""
    return base * 2 ** (attempt - 1) + random.uniform(0, base)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> FetchResult:
    """GET `url`, retrying 429/5xx/timeouts/connection errors up to `max_retries` times.

    Other 4xx are final (retrying a 404 just repeats the 404). A redirect loop
    is final too. `Retry-After` overrides the computed backoff.
    """
    attempt = 0
    t_start = time.perf_counter()
    while True:
        attempt += 1
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
                body = resp.content
        except (TimeoutError, httpx.TransportError) as exc:
            reason = type(exc).__name__
            if attempt <= max_retries:
                delay = backoff(attempt, base_delay)
                log.info("retry %s in %.2fs after %s (attempt %d)", url, delay, reason, attempt)
                await asyncio.sleep(delay)
                continue
            return FetchResult(url, url, 0, "", b"", time.perf_counter() - t_start, attempt, reason)
        except httpx.HTTPError as exc:  # e.g. TooManyRedirects: not transient
            reason = f"{type(exc).__name__}: {exc}"
            return FetchResult(url, url, 0, "", b"", time.perf_counter() - t_start, attempt, reason)
        if resp.status_code in RETRY_STATUSES and attempt <= max_retries:
            delay = retry_after_seconds(resp.headers.get("retry-after"))
            source = "Retry-After"
            if delay is None:
                delay, source = backoff(attempt, base_delay), "backoff"
            log.info(
                "retry %s in %.2fs (%s) after HTTP %d (attempt %d)",
                url,
                delay,
                source,
                resp.status_code,
                attempt,
            )
            await asyncio.sleep(delay)
            continue
        return FetchResult(
            url=url,
            final_url=str(resp.url),
            status=resp.status_code,
            content_type=resp.headers.get("content-type", ""),
            body=body,
            elapsed=time.perf_counter() - t_start,
            attempts=attempt,
            error=None if resp.is_success else f"HTTP {resp.status_code}",
        )
