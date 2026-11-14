"""Title, readable text and links from an HTML page (stdlib ``html.parser``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import override

SKIP = {"script", "style", "nav", "noscript", "template", "svg", "head"}
BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}


@dataclass
class Extracted:
    title: str = ""
    text: str = ""
    links: list[str] = field(default_factory=lambda: list[str]())


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: list[str] = []
        self.chunks: list[str] = []
        self.links: list[str] = []
        self._skip = 0
        self._in_title = False

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag == "title":
            self._in_title = True
        elif tag in SKIP:
            self._skip += 1
        if tag in BLOCK:
            self.chunks.append("\n")

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in SKIP and self._skip:
            self._skip -= 1
        if tag in BLOCK:
            self.chunks.append("\n")

    @override
    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.chunks.append(data)


def extract(html: str) -> Extracted:
    p = _Parser()
    p.feed(html)
    p.close()
    lines = (" ".join(line.split()) for line in "".join(p.chunks).splitlines())
    text = "\n\n".join(line for line in lines if line)
    return Extracted(" ".join("".join(p.title).split()), text, p.links)
