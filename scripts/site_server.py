"""A local mirror of the corpus as a small multi-host website — a crawl target we own.

    python scripts/site_server.py [--hosts 5] [--base-port 8101] [--latency 50:150]

Serves data/sections as HTML across N "hosts" (one port each, all 127.0.0.1),
with realistic friction for the Lab 6 crawler:

* every response waits 50–150 ms (configurable) — the network, simulated;
* ``/robots.txt`` disallows ``/private/`` (linked from every home page);
* ~1 in 20 section pages answers ``429 Too Many Requests`` + ``Retry-After: 1``
  the first time it is requested, ~1 in 30 answers ``503`` once;
* ``/old/...`` permanently redirects to the new URL.
"""

from __future__ import annotations

import argparse
import html
import random
import sys
import threading
import time
import zlib
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_gutenberg import BOOKS  # noqa: E402


class Site:
    def __init__(
        self, root: Path, n_hosts: int, base_port: int, latency: tuple[float, float]
    ) -> None:
        self.ports = [base_port + i for i in range(n_hosts)]
        self.latency = latency
        self.parts: dict[int, list[Path]] = defaultdict(list)
        for p in sorted(root.glob("pg*-*.txt")):
            book = int(p.stem[2:].split("-")[0])
            self.parts[book].append(p)
        self.books = sorted(self.parts)
        self.served: set[str] = set()
        self.lock = threading.Lock()

    def host_of(self, book: int) -> int:
        return self.ports[self.books.index(book) % len(self.ports)]

    def url(self, book: int, part: int | None = None) -> str:
        base = f"http://127.0.0.1:{self.host_of(book)}/book/{book}/"
        return base if part is None else f"{base}{part}"

    def first_time(self, key: str) -> bool:
        with self.lock:
            if key in self.served:
                return False
            self.served.add(key)
            return True


def page(title: str, body: str) -> bytes:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"</head><body><h1>{html.escape(title)}</h1>{body}</body></html>"
    ).encode()


def make_handler(site: Site, port: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "findex-mirror/1.0"

        def log_message(self, format: str, *args: object) -> None:  # quiet
            pass

        def send(
            self,
            status: int,
            body: bytes,
            ctype: str = "text/html; charset=utf-8",
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            time.sleep(random.uniform(*site.latency))
            path = self.path.split("?")[0]
            if path == "/robots.txt":
                return self.send(200, b"User-agent: *\nDisallow: /private/\n", "text/plain")
            if path.startswith("/private/"):
                return self.send(200, page("Private", "<p>Crawlers must not see this.</p>"))
            if path.startswith("/old/"):
                return self.send(301, b"", headers={"Location": path[4:]})
            if path == "/":
                return self.home()
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2 and parts[0] == "book" and parts[1].isdigit():
                book = int(parts[1])
                if book not in site.parts or site.host_of(book) != port:
                    return self.send(404, page("Not found", ""))
                if len(parts) == 2:
                    return self.book(book)
                if len(parts) == 3 and parts[2].isdigit():
                    return self.section(book, int(parts[2]))
            return self.send(404, page("Not found", ""))

        def home(self) -> None:
            mine = [b for b in site.books if site.host_of(b) == port]
            items = "".join(
                f"<li><a href='/book/{b}/'>{html.escape(BOOKS.get(b, str(b)))}</a></li>"
                for b in mine
            )
            others = "".join(
                f"<li><a href='http://127.0.0.1:{p}/'>mirror {p}</a></li>"
                for p in site.ports
                if p != port
            )
            body = (
                f"<ul>{items}</ul><h2>Other mirrors</h2><ul>{others}</ul>"
                "<p><a href='/private/admin'>admin</a></p>"
            )
            self.send(200, page(f"Library mirror {port}", body))

        def book(self, book: int) -> None:
            n = len(site.parts[book])
            items = "".join(f"<li><a href='{i}'>Part {i}</a></li>" for i in range(1, n + 1))
            self.send(
                200, page(BOOKS.get(book, str(book)), f"<a href='/'>home</a><ol>{items}</ol>")
            )

        def section(self, book: int, part: int) -> None:
            files = site.parts[book]
            if not 1 <= part <= len(files):
                return self.send(404, page("Not found", ""))
            key = f"{port}{self.path}"
            h = zlib.crc32(key.encode())
            if h % 20 == 0 and site.first_time("429" + key):
                return self.send(429, b"slow down", "text/plain", {"Retry-After": "1"})
            if h % 30 == 1 and site.first_time("503" + key):
                return self.send(503, b"try again", "text/plain")
            text = files[part - 1].read_text(encoding="utf-8")
            title, _, body = text.partition("\n")
            paras = "".join(f"<p>{html.escape(p)}</p>" for p in body.split("\n\n") if p.strip())
            nav = [f"<a href='/book/{book}/'>contents</a>"]
            if part > 1:
                nav.append(f"<a href='{part - 1}'>previous</a>")
            if part < len(files):
                nav.append(f"<a href='/old/book/{book}/{part + 1}'>next</a>")
            other = site.books[(site.books.index(book) + part) % len(site.books)]
            nav.append(f"<a href='{site.url(other, 1)}'>see also</a>")
            nav.append("<a href='#top'>top</a>")
            self.send(200, page(title, f"<nav>{' | '.join(nav)}</nav>{paras}"))

    return Handler


def serve(
    root: Path, n_hosts: int, base_port: int, latency: tuple[float, float]
) -> list[ThreadingHTTPServer]:
    site = Site(root, n_hosts, base_port, latency)
    servers: list[ThreadingHTTPServer] = []
    for port in site.ports:
        srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(site, port))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
    return servers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/sections"))
    ap.add_argument("--hosts", type=int, default=5)
    ap.add_argument("--base-port", type=int, default=8101)
    ap.add_argument("--latency", default="50:150", help="min:max milliseconds per response")
    args = ap.parse_args()
    lo, hi = (float(x) / 1000 for x in args.latency.split(":"))
    serve(args.root, args.hosts, args.base_port, (lo, hi))
    print(
        f"serving {args.root} on ports {args.base_port}..{args.base_port + args.hosts - 1}",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
