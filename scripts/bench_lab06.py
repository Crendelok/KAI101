"""Lab 6 benchmark: crawl the local mirror at concurrency 1 / 5 / 20.

    python scripts/bench_lab06.py [--max-pages 200]

For each configuration a *fresh* mirror server is started (so its one-time
429/503 responses happen again) and the crawl runs in its own process with
asyncio debug mode on; slow-callback warnings (blocking calls) are counted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def one(args: argparse.Namespace) -> None:
    from bench_lab05 import peak_rss_bytes

    from findex.crawler.crawl import CrawlStats, crawl

    slow: list[str] = []

    class Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "Executing" in record.getMessage():
                slow.append(record.getMessage())

    logging.getLogger("asyncio").addHandler(Catch())
    log_file = logging.FileHandler(args.log, mode="w", encoding="utf-8")
    log_file.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    for name in ("findex.crawl.urls", "findex.crawler.fetch"):
        lg = logging.getLogger(name)
        lg.addHandler(log_file)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    stats = CrawlStats()

    async def run() -> int:
        asyncio.get_running_loop().slow_callback_duration = 0.05
        n = 0
        with open(os.devnull, "w", encoding="utf-8") as sink:
            async for page in crawl(
                [args.seed],
                max_pages=args.max_pages,
                concurrency=args.concurrency,
                per_host=args.per_host,
                stats=stats,
            ):
                sink.write(json.dumps({"url": page.url, "text": page.text}) + "\n")
                n += 1
        return n

    t0 = time.perf_counter()
    pages = asyncio.run(run(), debug=args.debug)
    print(
        json.dumps(
            {
                "pages": pages,
                "wall": time.perf_counter() - t0,
                "errors": stats.errors,
                "retried": stats.retried,
                "robots": stats.robots_blocked,
                "peak_rss": peak_rss_bytes(),
                "slow_callbacks": len(slow),
                "slow_examples": slow[:4],
            }
        )
    )


def run_one(conc: int, per_host: int, max_pages: int, log: Path, debug: bool) -> dict[str, object]:
    """A fresh mirror server (so its one-time 429/503s recur) + one crawl in its own process."""
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "site_server.py")],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert server.stdout is not None
    server.stdout.readline()  # "serving ..."
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "one",
        "--concurrency",
        str(conc),
        "--per-host",
        str(per_host),
        "--max-pages",
        str(max_pages),
        "--log",
        str(log),
    ]
    try:
        out = subprocess.run(
            [*cmd, "--debug"] if debug else cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        )
    finally:
        server.terminate()
        server.wait()
    return json.loads(out.stdout.strip().splitlines()[-1])


def bench(args: argparse.Namespace) -> None:
    rows = []
    for conc, per_host in ((1, 2), (5, 2), (20, 2), (20, 4)):
        # Timing run (normal loop), then a separate debug-mode run that only counts
        # slow callbacks: debug mode's own overhead would distort the timings.
        runs = {}
        for debug in (False, True):
            log = ROOT / "docs" / f"crawl_c{conc}_h{per_host}.log"
            runs[debug] = run_one(conc, per_host, args.max_pages, log, debug)
        r = runs[False] | {"slow_callbacks": runs[True]["slow_callbacks"]}
        rows.append((conc, per_host, r))
        print(f"c={conc} per_host={per_host}: {r}", flush=True)
    print(
        "\n| Concurrency | Per host | Pages | Wall (s) | Pages/s | Errors | Retried | Peak RSS "
        "| Slow callbacks |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for conc, per_host, r in rows:
        print(
            f"| {conc} | {per_host} | {r['pages']} | {r['wall']:.1f} "
            f"| {r['pages'] / r['wall']:.1f} "
            f"| {r['errors']} | {r['retried']} | {r['peak_rss'] / 2**20:.0f} MiB "
            f"| {r['slow_callbacks']} |"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    o = sub.add_parser("one")
    o.add_argument("--seed", default="http://127.0.0.1:8101/")
    o.add_argument(
        "--debug", action="store_true", help="asyncio debug mode (counts slow callbacks)"
    )
    o.add_argument("--concurrency", type=int, required=True)
    o.add_argument("--per-host", type=int, required=True)
    o.add_argument("--max-pages", type=int, default=200)
    o.add_argument("--log", default="crawl.log")
    p.add_argument("--max-pages", type=int, default=200)
    args = p.parse_args()
    if args.cmd == "one":
        one(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()
