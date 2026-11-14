"""``robots.txt``, fetched once per origin and cached."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import fetch

log = logging.getLogger(__name__)


class RobotsCache:
    """``await robots.allowed(url)`` — per RFC 9309:

    * 2xx — obey the rules;
    * 4xx (no robots.txt) — everything is allowed;
    * 5xx / unreachable — assume everything is *disallowed* for that origin.
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self._client = client
        self._agent = user_agent
        self._rules: dict[str, RobotFileParser | bool] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _rules_for(self, url: str) -> RobotFileParser | bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._rules:
            return self._rules[origin]
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:  # many workers hit a new host at once: fetch robots.txt once
            if origin not in self._rules:
                res = await fetch(self._client, f"{origin}/robots.txt", max_retries=1)
                rules: RobotFileParser | bool
                if res.ok:
                    parser = RobotFileParser()
                    parser.parse(res.body.decode("utf-8", errors="replace").splitlines())
                    rules = parser
                elif 400 <= res.status < 500:
                    rules = True
                else:
                    log.warning(
                        "robots.txt unreachable for %s (%s): not crawling it", origin, res.error
                    )
                    rules = False
                self._rules[origin] = rules
            return self._rules[origin]

    async def allowed(self, url: str) -> bool:
        rules = await self._rules_for(url)
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(self._agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        rules = await self._rules_for(url)
        if isinstance(rules, bool):
            return None
        delay = rules.crawl_delay(self._agent)
        return float(delay) if delay is not None else None
