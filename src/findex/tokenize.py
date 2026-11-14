"""M2 — a streaming tokenizer.

Pipeline per document: NFC-normalize -> casefold -> ``re.finditer`` (lazy).

Token policy
------------
* **Letters & digits.** A token is a run of Unicode letters/digits (``\\w`` minus
  ``_``), so Latin, Cyrillic, Greek, CJK etc. all work.
* **Combining marks** (U+0300–U+036F) stay attached to the preceding letter, so a
  mark that has no precomposed NFC form doesn't split a word in two.
* **Apostrophes** *inside* a word are kept and unified to ASCII ``'``:
  ``don't``, ``o'clock``, ``п'ять``, ``пʼять`` (U+02BC) and ``don’t`` (U+2019)
  all produce the same token. Leading/trailing apostrophes (quotes) are dropped.
* **Hyphens split**: ``state-of-the-art`` -> ``state``, ``of``, ``the``, ``art``.
  Users search for the parts far more often than for the compound.
* **Digits are kept**, alone or mixed: ``2024``, ``python3``, ``3d``.
  ``3.14`` -> ``3``, ``14`` (punctuation always splits).
* **Underscore splits** (``snake_case`` -> ``snake``, ``case``).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_LETTER = r"[^\W_]"
_MARK = r"[̀-ͯ]"
_CHUNK = rf"{_LETTER}(?:{_LETTER}|{_MARK})*"
_APOSTROPHES = "'’ʼ"  # ASCII, right single quote, modifier letter apostrophe

TOKEN_RE = re.compile(rf"{_CHUNK}(?:[{_APOSTROPHES}]{_CHUNK})*")
_APOS_TABLE = str.maketrans({"’": "'", "ʼ": "'"})


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens from `text`, lazily."""
    for match in TOKEN_RE.finditer(normalize(text)):
        yield match.group().translate(_APOS_TABLE)
