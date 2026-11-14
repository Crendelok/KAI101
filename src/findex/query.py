"""The query language (Lab 3): a recursive-descent parser and a query tree.

Grammar (lowest precedence first)::

    query    := or_expr EOF
    or_expr  := and_expr ("OR" and_expr)*
    and_expr := unary (["AND"] unary)*          # juxtaposition is AND
    unary    := "NOT" unary | primary
    primary  := WORD | "quoted phrase" | "(" or_expr ")"

So AND binds tighter than OR: ``a OR b c`` is ``Or(a, And(b, c))``.
Nodes are frozen dataclasses (free ``__eq__``, so parser tests compare trees)
and overload ``&``, ``|``, ``~`` so trees can be built in code too.
"""

from __future__ import annotations

import re
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from findex.index import Index
from findex.setops import Engine, difference, intersect, union
from findex.tokenize import tokenize


class QueryError(ValueError):
    """A query that can't be parsed or evaluated."""


class Node:
    __slots__ = ()

    def __and__(self, other: Node) -> And:
        return And(self, other)

    def __or__(self, other: Node) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index, engine: Engine = "merge") -> Sequence[int]:
        raise NotImplementedError

    def positive_terms(self) -> Iterator[str]:
        """Terms that contribute to ranking (everything not under a NOT)."""
        return iter(())


@dataclass(frozen=True, slots=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index, engine: Engine = "merge") -> Sequence[int]:
        return index[self.text].doc_ids if self.text in index else array("I")

    def positive_terms(self) -> Iterator[str]:
        yield self.text

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index, engine: Engine = "merge") -> Sequence[int]:
        if not self.words:
            return array("I")
        if any(w not in index for w in self.words):
            return array("I")
        candidates: Sequence[int] = index[self.words[0]].doc_ids
        for w in self.words[1:]:
            candidates = intersect(candidates, index[w].doc_ids, engine)
        if len(self.words) == 1:
            return candidates
        if not index.has_positions:
            raise QueryError("phrase queries need an index built with --positions")
        lists = [index[w] for w in self.words]
        out = array("I")
        for doc_id in candidates:
            starts = set(lists[0].positions_at(lists[0].find(doc_id)))
            for offset, pl in enumerate(lists[1:], start=1):
                shifted = {p - offset for p in pl.positions_at(pl.find(doc_id))}
                starts &= shifted
                if not starts:
                    break
            if starts:
                out.append(doc_id)
        return out

    def positive_terms(self) -> Iterator[str]:
        yield from self.words

    def __str__(self) -> str:
        return '"' + " ".join(self.words) + '"'


@dataclass(frozen=True, slots=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> Sequence[int]:
        # a AND NOT b is a difference, never "everything minus b" first.
        if isinstance(self.right, Not):
            return difference(
                self.left.evaluate(index, engine), self.right.child.evaluate(index, engine), engine
            )
        if isinstance(self.left, Not):
            return difference(
                self.right.evaluate(index, engine), self.left.child.evaluate(index, engine), engine
            )
        return intersect(
            self.left.evaluate(index, engine), self.right.evaluate(index, engine), engine
        )

    def positive_terms(self) -> Iterator[str]:
        yield from self.left.positive_terms()
        yield from self.right.positive_terms()

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True, slots=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> Sequence[int]:
        return union(self.left.evaluate(index, engine), self.right.evaluate(index, engine), engine)

    def positive_terms(self) -> Iterator[str]:
        yield from self.left.positive_terms()
        yield from self.right.positive_terms()

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True, slots=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> Sequence[int]:
        return difference(index.all_doc_ids(), self.child.evaluate(index, engine), engine)

    def __str__(self) -> str:
        return f"NOT {self.child}"


# --- lexer + parser -----------------------------------------------------------------

_LEX = re.compile(r'"(?P<phrase>[^"]*)"?|(?P<lp>\()|(?P<rp>\))|(?P<word>[^\s()"]+)')
_OPS = {"AND", "OR", "NOT"}


def _lex(query: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in _LEX.finditer(query):
        kind = m.lastgroup
        assert kind is not None
        value = m.group(kind) if kind != "phrase" else m.group("phrase")
        if kind == "word" and value in _OPS:
            kind = value
        out.append((kind, value or ""))
    return out


class _Parser:
    def __init__(self, query: str) -> None:
        self.toks = _lex(query)
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        if not self.toks:
            raise QueryError("empty query")
        node = self.or_expr()
        if self.peek() is not None:
            raise QueryError(f"unexpected {self.toks[self.i][1]!r}")
        if node is None:
            raise QueryError("query has no searchable terms")
        return node

    def or_expr(self) -> Node | None:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            right = self.and_expr()
            node = right if node is None else node if right is None else Or(node, right)
        return node

    def and_expr(self) -> Node | None:
        node = self.unary()
        while self.peek() not in (None, "OR", "rp"):
            if self.peek() == "AND":
                self.take()
            right = self.unary()
            node = right if node is None else node if right is None else And(node, right)
        return node

    def unary(self) -> Node | None:
        if self.peek() == "NOT":
            self.take()
            child = self.unary()
            if child is None:
                raise QueryError("NOT needs something to negate")
            return Not(child)
        return self.primary()

    def primary(self) -> Node | None:
        kind = self.peek()
        if kind is None:
            raise QueryError("query ends too early")
        if kind == "lp":
            self.take()
            node = self.or_expr()
            if self.peek() != "rp":
                raise QueryError("missing ')'")
            self.take()
            return node
        if kind == "rp":
            raise QueryError("unbalanced ')'")
        if kind in ("AND", "OR"):
            raise QueryError(f"{kind} needs a term on both sides")
        _, value = self.take()
        words = tuple(tokenize(value))
        if kind == "phrase":
            return Phrase(words) if words else None
        if not words:
            return None  # pure punctuation: ignore
        # A hyphenated word ("state-of-the-art") is a phrase of its parts.
        return Term(words[0]) if len(words) == 1 else Phrase(words)


def parse(query: str) -> Node:
    """Parse a query string into a tree. Raises QueryError on bad syntax."""
    return _Parser(query).parse()
