"""URL canonicalization: one spelling per page, so the `seen` set actually dedupes."""

from __future__ import annotations

import posixpath
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Absolute, normalized http(s) URL — or None for anything we won't crawl.

    * resolve against `base`; only ``http``/``https``
    * scheme and host lowercased; default ports dropped
    * fragment stripped (``#top`` is the same page)
    * empty path -> ``/``; ``.``/``..`` segments resolved; ``//`` collapsed
    * query parameters sorted (``?b=2&a=1`` == ``?a=1&b=2``)

    The trailing slash is *kept* in the URL (``/book/1/`` and ``/book/1`` resolve
    relative links differently), but ignored by :func:`seen_key`.
    """
    try:
        absolute = urljoin(base, url.strip()) if base else url.strip()
        parts = urlsplit(absolute)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower()
    try:
        port = parts.port
    except ValueError:
        return None
    netloc = host if port in (None, DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    path = parts.path or "/"
    trailing = path.endswith("/")
    path = posixpath.normpath(path)
    if path in (".", "//"):
        path = "/"
    if trailing and not path.endswith("/"):
        path += "/"
    path = path.replace("//", "/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def seen_key(url: str) -> str:
    """Dedup key: the canonical URL with any trailing slash removed (except the root)."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def host_of(url: str) -> str:
    """``host:port`` — the unit for per-host politeness."""
    return urlsplit(url).netloc


def domain_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()
