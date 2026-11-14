"""Split data/gutenberg/pg*.txt into ~1,500-word sections under data/sections/.

    python scripts/make_sections.py [--words 1500]

Each section is one document. Cuts happen only at paragraph boundaries, and the
first line of every section is its title ("Moby-Dick — part 12/143"), which
`iter_documents` picks up as the document title.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_gutenberg import BOOKS  # noqa: E402

PARA = re.compile(r"\n\s*\n")


def sections(text: str, words: int) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for para in PARA.split(text):
        para = para.strip()
        if not para:
            continue
        buf.append(para)
        n += len(para.split())
        if n >= words:
            out.append("\n\n".join(buf))
            buf, n = [], 0
    if buf:
        if out and n < words // 3:  # don't leave a tiny tail section
            out[-1] += "\n\n" + "\n\n".join(buf)
        else:
            out.append("\n\n".join(buf))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=1500)
    args = ap.parse_args()
    src, dst = Path("data/gutenberg"), Path("data/sections")
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.txt"):
        old.unlink()
    total = 0
    for path in sorted(src.glob("pg*.txt")):
        book_id = int(path.stem[2:])
        title = BOOKS.get(book_id, path.stem)
        parts = sections(path.read_text(encoding="utf-8"), args.words)
        for i, part in enumerate(parts, start=1):
            header = f"{title} — part {i}/{len(parts)}"
            (dst / f"{path.stem}-{i:04d}.txt").write_text(f"{header}\n\n{part}\n", encoding="utf-8")
        total += len(parts)
    print(f"wrote {total} sections to {dst}")


if __name__ == "__main__":
    main()
