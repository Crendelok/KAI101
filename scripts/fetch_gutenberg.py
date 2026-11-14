"""Download public-domain books from Project Gutenberg into data/gutenberg/.

    python scripts/fetch_gutenberg.py              # the default list below
    python scripts/fetch_gutenberg.py 1342 2701    # specific ebook IDs

One book = one .txt document. The Gutenberg licence header/footer is stripped.
Be polite: one request per second, cached files are skipped.
Then run scripts/make_sections.py to split books into ~1,500-word sections.
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

BOOKS: dict[int, str] = {
    1342: "Pride and Prejudice",
    84: "Frankenstein",
    11: "Alice's Adventures in Wonderland",
    2701: "Moby-Dick",
    1661: "The Adventures of Sherlock Holmes",
    98: "A Tale of Two Cities",
    74: "The Adventures of Tom Sawyer",
    76: "Adventures of Huckleberry Finn",
    345: "Dracula",
    174: "The Picture of Dorian Gray",
    1952: "The Yellow Wallpaper",
    2600: "War and Peace",
    5200: "Metamorphosis",
    1080: "A Modest Proposal",
    4300: "Ulysses",
    2554: "Crime and Punishment",
    1260: "Jane Eyre",
    768: "Wuthering Heights",
    158: "Emma",
    205: "Walden",
    120: "Treasure Island",
    1400: "Great Expectations",
    219: "Heart of Darkness",
    36: "The War of the Worlds",
    135: "Les Misérables",
    2591: "Grimms' Fairy Tales",
    1232: "The Prince",
    3207: "Leviathan",
    16: "Peter Pan",
    55: "The Wonderful Wizard of Oz",
    1184: "The Count of Monte Cristo",
    2542: "A Doll's House",
    844: "The Importance of Being Earnest",
    6130: "The Iliad",
    1998: "Thus Spake Zarathustra",
    408: "The Souls of Black Folk",
    2814: "Dubliners",
    1727: "The Odyssey",
    996: "Don Quixote",
    1497: "The Republic",
}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def strip_boilerplate(text: str) -> str:
    if m := START.search(text):
        text = text[m.end() :]
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
    main([int(a) for a in sys.argv[1:]] or list(BOOKS))
