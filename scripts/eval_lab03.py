"""Lab 3: ranking sanity checks on the real corpus + precision@5 for TF-IDF vs BM25.

python scripts/eval_lab03.py index.fidx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from findex.index import Index, Posting
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import search
from findex.store import open_index

QUERIES = Path(__file__).parent.parent / "eval" / "queries.json"


def book(title: str) -> str:
    return title.split(" — part ")[0]


def precision_at_5(index: Index, scorer: Scorer, q: str, books: list[str]) -> float:
    hits = search(index, " OR ".join(q.split()), scorer, k=5)  # ranked "any of these words"
    return sum(book(h.title) in books for h in hits) / 5


def sanity(index: Index) -> None:
    bm25 = BM25()
    print("### Ranking sanity checks (BM25, real corpus)\n")
    # 1. rare vs common
    rare, common = "harpoon", "sea"
    print(
        f"1. `{rare}` df={index.df(rare)}, `{common}` df={index.df(common)}; idf "
        f"{bm25.idf(index.df(rare), index.num_docs):.2f} vs "
        f"{bm25.idf(index.df(common), index.num_docs):.2f}."
    )
    res = search(index, f"{rare} OR {common}", bm25, k=5)
    has_rare = [index[rare].find(r.doc_id) >= 0 for r in res]
    print(f"   Top 5 for `{rare} OR {common}`: {sum(has_rare)}/5 contain `{rare}`.")
    # 2. saturation
    avg = index.avg_doc_length
    w = [bm25.tf_weight(tf, int(avg), avg) for tf in (1, 2, 5, 10, 19, 20)]
    print(
        f"2. BM25 tf weight at avg length for tf=1,2,5,10,19,20: "
        f"{', '.join(f'{x:.3f}' for x in w)} — the 20th repetition adds {w[5] - w[4]:.4f} "
        f"({(w[5] - w[4]) / w[0]:.1%} of the first)."
    )
    # 3. short vs long doc, one match each
    term = "telescope"
    pl = index[term]
    ones = [(index.doc_length(d), d) for d, tf in zip(pl.doc_ids, pl.tfs, strict=True) if tf == 1]
    (short_len, short), (long_len, long_) = min(ones), max(ones)
    s_short = bm25.score(term, Posting(short, 1), index)
    s_long = bm25.score(term, Posting(long_, 1), index)
    print(
        f"3. `{term}` once in a {short_len:,}-token section scores {s_short:.3f}; "
        f"once in a {long_len:,}-token section scores {s_long:.3f}.\n"
    )


def evaluate(index: Index) -> None:
    data = json.loads(QUERIES.read_text(encoding="utf-8"))
    print("### precision@5\n")
    print("| Query | Relevant book | TF-IDF | BM25 |")
    print("|---|---|---|---|")
    totals = {"tfidf": 0.0, "bm25": 0.0}
    for item in data["queries"]:
        p_t = precision_at_5(index, TfIdf(), item["q"], item["books"])
        p_b = precision_at_5(index, BM25(), item["q"], item["books"])
        totals["tfidf"] += p_t
        totals["bm25"] += p_b
        print(f"| {item['q']} | {', '.join(item['books'])} | {p_t:.1f} | {p_b:.1f} |")
    n = len(data["queries"])
    print(f"| **mean** | | **{totals['tfidf'] / n:.2f}** | **{totals['bm25'] / n:.2f}** |")


def main() -> None:
    with open_index(sys.argv[1] if len(sys.argv) > 1 else "index.fidx") as index:
        sanity(index)
        evaluate(index)


if __name__ == "__main__":
    main()
