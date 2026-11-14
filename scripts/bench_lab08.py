"""Lab 8 measurements.

python scripts/bench_lab08.py profile      # cProfile of build + search -> docs/profile_*.txt
python scripts/bench_lab08.py speed        # Lab 3 scorer vs NumPy scorer table
python scripts/bench_lab08.py eval         # precision@5: keyword / semantic / hybrid, 15 queries
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index.fidx"
QUERY3 = "whale ship captain"


def profile(_: list[str]) -> None:
    from findex.corpus import iter_documents
    from findex.index import build_index
    from findex.search import _matching_ids, search
    from findex.store import load

    docs = ROOT / "docs"
    prof = cProfile.Profile()
    prof.enable()
    build_index(iter_documents(ROOT / "data" / "sections"), positions=True)
    prof.disable()
    _dump(prof, docs / "profile_index.txt", "findex index data/sections (serial, positions)")

    index = load(INDEX)
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(50):
        _matching_ids.cache_clear()
        search(index, f"{QUERY3.replace(' ', ' OR ')}", k=10)
    prof.disable()
    _dump(prof, docs / "profile_search.txt", f"50 × search('{QUERY3.replace(' ', ' OR ')}')")


def _dump(prof: cProfile.Profile, path: Path, title: str) -> None:
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).strip_dirs().sort_stats("tottime")
    buf.write(f"# {title}\n# sorted by tottime (time spent in the function itself)\n")
    stats.print_stats(15)
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())


def ms(fn, *args, n: int = 20, **kw) -> float:  # type: ignore[no-untyped-def]
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(*args, **kw)
        best = min(best, time.perf_counter() - t)
    return best * 1000


def speed(_: list[str]) -> None:
    from findex.fast import search_fast
    from findex.scoring import BM25
    from findex.search import matching_ids, search
    from findex.store import load

    index = load(INDEX)
    rare = next(t for t in sorted(index, key=index.df) if index.df(t) == 3)
    cases = [
        ("common term", "the"),
        ("mid term", "whale"),
        ("rare term", rare),
        ("3-term query (OR)", "whale OR ship OR captain"),
        ("3-term query (AND)", "whale ship captain"),
    ]
    print("| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |")
    print("|---|---|---|---|---|")
    for label, q in cases:
        ids, _ = matching_ids(index, q)  # warm the query cache: both sides measure scoring
        slow = ms(search, index, q, BM25(), 10)
        fast = ms(search_fast, index, q, BM25(), 10)
        print(f"| {label} `{q}` | {len(ids):,} | {slow:.2f} | {fast:.2f} | {slow / fast:.1f}× |")


def evaluate(_: list[str]) -> None:
    from findex.semantic import SemanticIndex, search_mode
    from findex.store import load

    index = load(INDEX)
    sem = SemanticIndex.load(INDEX.with_suffix(".emb.npy"))
    data = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))
    items = [(q, "original") for q in data["queries"]] + [
        (q, "paraphrase") for q in data["paraphrases"]
    ]

    def book(title: str) -> str:
        return title.split(" — part ")[0]

    print("| Query | Kind | Keyword | Semantic | Hybrid |")
    print("|---|---|---|---|---|")
    totals: dict[str, dict[str, float]] = {"original": {}, "paraphrase": {}}
    for item, kind in items:
        row = []
        for mode in ("keyword", "semantic", "hybrid"):
            q = item["q"] if mode != "keyword" else " OR ".join(item["q"].split())
            res = search_mode(index, sem, q, mode=mode, k=5)  # type: ignore[arg-type]
            p = sum(book(h.title) in item["books"] for h in res) / 5
            totals[kind][mode] = totals[kind].get(mode, 0.0) + p
            row.append(p)
        print(f"| {item['q']} | {kind} | " + " | ".join(f"{p:.1f}" for p in row) + " |")
    for kind, n in (("original", len(data["queries"])), ("paraphrase", len(data["paraphrases"]))):
        t = totals[kind]
        print(
            f"| **mean, {kind}** | | **{t['keyword'] / n:.2f}** | **{t['semantic'] / n:.2f}** "
            f"| **{t['hybrid'] / n:.2f}** |"
        )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "speed"
    {"profile": profile, "speed": speed, "eval": evaluate}[cmd](sys.argv[2:])
