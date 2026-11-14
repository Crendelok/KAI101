"""Snippets (Lab 3): a ±80-character window around the best match, with spans to highlight."""

from __future__ import annotations

import html
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from findex.tokenize import find_tokens

RADIUS = 80


@dataclass(frozen=True, slots=True)
class Snippet:
    text: str
    spans: tuple[tuple[int, int], ...] = field(default=())  # (start, end) into `text`

    def highlight(self, before: str, after: str, escape: Callable[[str], str] = lambda s: s) -> str:
        out: list[str] = []
        pos = 0
        for start, end in self.spans:
            out.append(escape(self.text[pos:start]))
            out.append(before + escape(self.text[start:end]) + after)
            pos = end
        out.append(escape(self.text[pos:]))
        return "".join(out)

    def html(self) -> str:
        return self.highlight("<mark>", "</mark>", html.escape)

    def rich(self) -> str:
        from rich.markup import escape

        return self.highlight("[bold yellow]", "[/]", escape)

    def plain(self) -> str:
        return self.highlight("«", "»")


def make_snippet(text: str, terms: Collection[str], radius: int = RADIUS) -> Snippet:
    """Pick the window (centred on a match) that contains the most distinct query terms."""
    if not text:
        return Snippet("")
    terms = set(terms)
    matches = [(s, e, tok) for tok, s, e in find_tokens(text) if tok in terms]
    if not matches:
        end = min(len(text), 2 * radius)
        return Snippet(_squash(text[:end]) + ("…" if end < len(text) else ""))
    best_center, best_score = matches[0][0], -1
    lo = 0
    for s, _e, _tok in matches:  # sliding window over match positions
        while matches[lo][0] < s - radius:
            lo += 1
        hi = lo
        seen: set[str] = set()
        while hi < len(matches) and matches[hi][0] <= s + radius:
            seen.add(matches[hi][2])
            hi += 1
        if len(seen) > best_score:
            best_center, best_score = s, len(seen)
    start = max(0, best_center - radius)
    end = min(len(text), best_center + radius)
    # snap to word boundaries
    while start > 0 and not text[start - 1].isspace() and best_center - start < radius + 15:
        start -= 1
    while end < len(text) and not text[end].isspace() and end - best_center < radius + 15:
        end += 1
    window = text[start:end]
    spans = tuple((s - start, e - start) for s, e, _ in matches if s >= start and e <= end)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    squashed, spans = _squash_with_spans(window, spans)
    shift = len(prefix)
    return Snippet(prefix + squashed + suffix, tuple((s + shift, e + shift) for s, e in spans))


def _squash(s: str) -> str:
    return " ".join(s.split())


def _squash_with_spans(
    s: str, spans: tuple[tuple[int, int], ...]
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Collapse whitespace runs to one space while keeping spans pointing at the same text."""
    mapping: list[int] = []
    out: list[str] = []
    prev_space = True
    for ch in s:
        if ch.isspace():
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
        mapping.append(len(out))
    text = "".join(out).rstrip()

    def at(i: int) -> int:
        return min(mapping[i - 1] if i > 0 else 0, len(text))

    new_spans = tuple((at(st + 1) - 1, at(en)) for st, en in spans)
    return text, new_spans
