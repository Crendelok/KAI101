"""Lab 5 benchmark: serial vs threads vs processes (and free-threaded 3.13t).

    python scripts/bench_lab05.py cpu  [--runs 3] [--python PATH --label NAME --executors threads]
    python scripts/bench_lab05.py io                 # threads on an I/O-bound job
    python scripts/bench_lab05.py report             # table + docs/lab05_speedup.png

Every measured build runs in a *fresh subprocess* (``one``) so peak RSS is that
run's own peak and nothing is warm from the previous configuration. Results
accumulate in docs/lab05_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "lab05_results.json"
WORKERS = sorted({1, 2, 4, 8, os.cpu_count() or 1})


def peak_rss_bytes() -> int:
    """Peak resident set of *this* process, without third-party packages."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [  # noqa: RUF012
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError()
        return int(pmc.PeakWorkingSetSize)
    import resource

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb if sys.platform == "darwin" else kb * 1024


def one(args: argparse.Namespace) -> None:
    from findex.parallel import build_parallel

    _, rep = build_parallel(args.corpus, workers=args.workers, executor=args.executor)
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        json.dumps(
            {
                "wall": rep.wall,
                "build": rep.build,
                "merge": rep.merge,
                "cpu": rep.cpu,
                "peak_rss": peak_rss_bytes(),
                "gil": gil,
                "python": platform.python_version(),
            }
        )
    )


def run_config(python: str, corpus: str, executor: str, workers: int) -> dict[str, float]:
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "src"), PYTHON_GIL=os.environ.get("PYTHON_GIL", "")
    )
    if not env["PYTHON_GIL"]:
        env.pop("PYTHON_GIL")
    cmd = [
        python,
        str(Path(__file__).resolve()),
        "one",
        "--corpus",
        corpus,
        "--executor",
        executor,
        "--workers",
        str(workers),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=ROOT)
    return json.loads(out.stdout.strip().splitlines()[-1])


def load_results() -> list[dict[str, object]]:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else []


def cpu(args: argparse.Namespace) -> None:
    results = load_results()
    python = args.python or sys.executable
    configs = [("serial", 1)] if "serial" in args.executors else []
    configs += [(e, w) for e in ("threads", "processes") if e in args.executors for w in WORKERS]
    for executor, workers in configs:
        runs = []
        for i in range(args.runs):
            r = run_config(python, args.corpus, executor, workers)
            runs.append(r)
            print(
                f"{args.label:>10} {executor:>9} ×{workers:<2} run {i + 1}: wall {r['wall']:.2f}s "
                f"cpu {r['cpu']:.1f}s merge {r['merge']:.2f}s rss {r['peak_rss'] / 2**20:.0f} MiB "
                f"gil={r['gil']}",
                flush=True,
            )
        results = [
            x
            for x in results
            if not (
                x["label"] == args.label and x["executor"] == executor and x["workers"] == workers
            )
        ]
        results.append(
            {
                "label": args.label,
                "executor": executor,
                "workers": workers,
                "python": runs[0]["python"],
                "gil": runs[0]["gil"],
                "runs": runs,
            }
        )
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))


def io_bench(args: argparse.Namespace) -> None:
    """Fetch 50 pages from the local mirror (100 ms latency each) with N threads."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT / "scripts"))
    from site_server import serve

    servers = serve(ROOT / "data" / "sections", 1, 8150, (0.1, 0.1))
    urls = ["http://127.0.0.1:8150/robots.txt"] * 50  # every request waits 100 ms server-side

    def get(url: str) -> int:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return len(r.read())
        except OSError:
            return 0

    rows = []
    for n in (1, 2, 4, 8, 16, 50):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(n) as pool:
            list(pool.map(get, urls))
        rows.append((n, time.perf_counter() - t0))
    for s in servers:
        s.shutdown()
    base = rows[0][1]
    print("| Threads | Wall (s) | Speedup |\n|---|---|---|")
    for n, t in rows:
        print(f"| {n} | {t:.2f} | {base / t:.1f}× |")


def report(args: argparse.Namespace) -> None:
    results = load_results()
    serial = next(r for r in results if r["executor"] == "serial")
    base = statistics.median(x["wall"] for x in serial["runs"])  # type: ignore[index]
    print(
        "| Executor | Python | Workers | Wall (s, median) | CPU (s) | Peak RSS (parent) "
        "| Merge (s) | Speedup |"
    )
    print("|---|---|---|---|---|---|---|---|")
    series: dict[str, list[tuple[int, float]]] = {}
    for r in results:
        runs = r["runs"]
        wall = statistics.median(x["wall"] for x in runs)  # type: ignore[union-attr]
        cpu_s = statistics.median(x["cpu"] for x in runs)  # type: ignore[union-attr]
        rss = statistics.median(x["peak_rss"] for x in runs) / 2**20  # type: ignore[union-attr]
        merge = statistics.median(x["merge"] for x in runs)  # type: ignore[union-attr]
        name = f"{r['executor']}" + ("" if r["label"] == "gil" else f" ({r['label']})")
        py = f"{r['python']}{'' if r['gil'] else 't'}"
        print(
            f"| {name} | {py} | {r['workers']} | {wall:.2f} | {cpu_s:.1f} | {rss:.0f} MiB "
            f"| {merge:.2f} | {base / wall:.2f}× |"
        )
        if r["executor"] != "serial":
            series.setdefault(name, []).append((int(r["workers"]), base / wall))  # type: ignore[arg-type]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.plot(WORKERS, WORKERS, "--", color="#999", label="ideal (linear)")
    for name, pts in series.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs serial")
    ax.set_xticks(WORKERS)
    ax.set_title(f"Parallel indexing, {platform.processor() or platform.machine()}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = ROOT / "docs" / "lab05_speedup.png"
    fig.savefig(out)
    print(f"\nplot: {out.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one")
    o.add_argument("--corpus", default="data/sections")
    o.add_argument("--executor", required=True)
    o.add_argument("--workers", type=int, required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--corpus", default="data/sections")
    c.add_argument("--runs", type=int, default=3)
    c.add_argument("--python", default=None)
    c.add_argument("--label", default="gil")
    c.add_argument("--executors", nargs="+", default=["serial", "threads", "processes"])
    sub.add_parser("io")
    sub.add_parser("report")
    args = p.parse_args()
    {"one": one, "cpu": cpu, "io": io_bench, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
