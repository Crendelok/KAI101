"""Ranking functions (Lab 3) behind a structural ``Scorer`` protocol (Lab 4).

Anything with a matching ``score`` method *is* a Scorer — no inheritance, no
registration. Both implementations below are tiny classes; a plain object with
a ``score`` method would do just as well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from findex.index import Index, Posting

type ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """``(1 + ln tf) * ln(N / df)`` — log-scaled tf, no length normalization."""

    def idf(self, df: int, n_docs: int) -> float:
        return math.log(n_docs / df) if df else 0.0

    def tf_weight(self, tf: int, doc_len: int, avg_len: float) -> float:
        return 1.0 + math.log(tf) if tf > 0 else 0.0

    def score(self, term: str, posting: Posting, index: Index) -> float:
        return self.tf_weight(posting.tf, 0, 0.0) * self.idf(index.df(term), index.num_docs)


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25.

    * ``k1`` — tf saturation: how quickly extra repetitions stop helping.
    * ``b``  — length normalization: 1 = fully relative to the average length,
      0 = document length ignored.
    """

    k1: float = 1.5
    b: float = 0.75

    def idf(self, df: int, n_docs: int) -> float:
        # The "+1" form never goes negative for very common terms.
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def tf_weight(self, tf: int, doc_len: int, avg_len: float) -> float:
        norm = 1.0 - self.b + self.b * (doc_len / avg_len if avg_len else 1.0)
        return tf * (self.k1 + 1.0) / (tf + self.k1 * norm)

    def score(self, term: str, posting: Posting, index: Index) -> float:
        tfw = self.tf_weight(posting.tf, index.doc_length(posting.doc_id), index.avg_doc_length)
        return tfw * self.idf(index.df(term), index.num_docs)


def get_scorer(name: ScorerName) -> TfIdf | BM25:
    return BM25() if name == "bm25" else TfIdf()
