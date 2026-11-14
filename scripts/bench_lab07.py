"""Lab 7 measurements.

    python scripts/bench_lab07.py syncasync     # `def` vs `async def` /search, 20 concurrent
    python scripts/bench_lab07.py load          # oha (if on PATH) load tests -> table
    python scripts/bench_lab07.py load --url https://your-app.onrender.com --concurrency 20

Each local configuration starts its own `findex serve` process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
QUERY = "the OR and OR of"  # matches every document: the most CPU per request
LOAD_QUERY = '"white whale" OR captain'


@contextmanager
def server(port: int, *extra: str) -> Iterator[str]:
    findex = shutil.which("findex", path=str(Path(sys.executable).parent)) or "findex"
    proc = subprocess.Popen(
        [findex, "serve", "--index", str(ROOT / "index.fidx"), "--port", str(port), *extra],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("server did not become healthy")
        yield base
    finally:
        proc.terminate()
        proc.wait()


async def burst(base: str, n: int = 20) -> dict[str, float]:
    """n concurrent /search requests while a probe hits /health every 10 ms."""
    async with httpx.AsyncClient(base_url=base, timeout=60) as c:
        await c.get("/search", params={"q": QUERY})  # warm up
        probes: list[float] = []
        stop = asyncio.Event()

        async def probe() -> None:
            while not stop.is_set():
                t = time.perf_counter()
                await c.get("/health")
                probes.append((time.perf_counter() - t) * 1000)
                await asyncio.sleep(0.01)

        async def one() -> float:
            t = time.perf_counter()
            r = await c.get("/search", params={"q": QUERY, "k": 10})
            r.raise_for_status()
            return (time.perf_counter() - t) * 1000

        p = asyncio.create_task(probe())
        await asyncio.sleep(0.05)
        t0 = time.perf_counter()
        lat = await asyncio.gather(*(one() for _ in range(n)))
        wall = time.perf_counter() - t0
        stop.set()
        await p
    return {
        "wall_s": wall,
        "search_p50": statistics.median(lat),
        "search_max": max(lat),
        "health_p50": statistics.median(probes),
        "health_max": max(probes),
    }


def syncasync(_: argparse.Namespace) -> None:
    print(
        "| /search defined as | 20 concurrent: wall | search p50 | search max "
        "| /health p50 during burst | /health max |"
    )
    print("|---|---|---|---|---|---|")
    for label, extra in (
        ("`def` (threadpool)", ()),
        ("`async def` (event loop)", ("--search-async",)),
    ):
        runs = []
        with server(8210, *extra) as base:
            for _ in range(3):
                runs.append(asyncio.run(burst(base)))
        r = {k: statistics.median(x[k] for x in runs) for k in runs[0]}
        print(
            f"| {label} | {r['wall_s']:.2f} s | {r['search_p50']:.0f} ms "
            f"| {r['search_max']:.0f} ms "
            f"| {r['health_p50']:.1f} ms | {r['health_max']:.0f} ms |"
        )


def oha(url: str, concurrency: int, seconds: int) -> dict[str, float]:
    exe = shutil.which("oha")
    if exe is None:
        raise SystemExit("oha not found on PATH (winget install hatoo.oha / cargo install oha)")
    out = subprocess.run(
        [
            exe,
            "-z",
            f"{seconds}s",
            "-c",
            str(concurrency),
            "--no-tui",
            "--output-format",
            "json",
            url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    pct = data["latencyPercentiles"]
    return {
        "rps": data["summary"]["requestsPerSec"],
        "p50": pct["p50"] * 1000,
        "p95": pct["p95"] * 1000,
        "p99": pct["p99"] * 1000,
        "ok": data["summary"]["successRate"],
    }


def load(args: argparse.Namespace) -> None:
    from urllib.parse import quote

    path = f"/search?q={quote(LOAD_QUERY)}&k=10"
    rows = []
    if args.url:
        rows.append(
            (
                "deployed",
                args.concurrency,
                oha(args.url.rstrip("/") + path, args.concurrency, args.seconds),
            )
        )
    else:
        configs = [
            ("local, 1 worker, `async def`", ("--search-async",)),
            ("local, 1 worker, `def`", ()),
            ("local, 4 workers, `def`", ("--workers", "4")),
        ]
        for label, extra in configs:
            with server(8220, *extra) as base:
                oha(base + path, 10, 2)  # warm-up (and every worker's cache)
                rows.append((label, 50, oha(base + path, 50, args.seconds)))
    print("| Target | Route | Concurrency | RPS | p50 (ms) | p95 | p99 | Success |")
    print("|---|---|---|---|---|---|---|---|")
    for label, conc, r in rows:
        print(
            f"| {label} | `/search` | {conc} | {r['rps']:.0f} | {r['p50']:.1f} | {r['p95']:.1f} "
            f"| {r['p99']:.1f} | {r['ok']:.0%} |"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("syncasync")
    lp = sub.add_parser("load")
    lp.add_argument("--url", default=None)
    lp.add_argument("--concurrency", type=int, default=20)
    lp.add_argument("--seconds", type=int, default=10)
    args = p.parse_args()
    os.chdir(ROOT)
    {"syncasync": syncasync, "load": load}[args.cmd](args)


if __name__ == "__main__":
    main()
