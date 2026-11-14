"""Download public-domain books from Project Gutenberg into data/gutenberg/.

    python scripts/fetch_gutenberg.py              # the default list below
    python scripts/fetch_gutenberg.py 1342 2701    # specific ebook IDs

One book = one .txt document. The Gutenberg licence header/footer is stripped.
Be polite: one request per second, cached files are skipped.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

try:  # verify TLS against the OS certificate store (antivirus / corporate proxies)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

DEFAULT_IDS = [
    1342, 84, 11, 2701, 1661, 98, 74, 76, 345, 174, 1952, 2600, 5200, 1080, 4300,
    2554, 1260, 768, 158, 205, 120, 1400, 219, 36, 135, 2591, 1232, 3207, 16, 55,
    1184, 2542, 844, 6130, 1998, 408, 2814, 1727, 996, 30254,
]
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def strip_boilerplate(text: str) -> str:
    if m := START.search(text):
        text = text[m.end():]
    if m := END.search(text):
        text = text[: m.start()]
    return text.strip()


def main(ids: list[int]) -> None:
    out = Path("data/gutenberg")
    out.mkdir(parents=True, exist_ok=True)
    for book_id in ids:
        dest = out / f"pg{book_id}.txt"
        if dest.exists():
            continue
        req = urllib.request.Request(URL.format(id=book_id), headers={"User-Agent": "findex-lab"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except OSError as exc:
            print(f"skip {book_id}: {exc}", file=sys.stderr)
            continue
        dest.write_text(strip_boilerplate(raw), encoding="utf-8")
        print(f"saved {dest} ({dest.stat().st_size // 1024} KiB)")
        time.sleep(1)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or DEFAULT_IDS)
