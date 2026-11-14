# Lab 8 — hotspots, written down before changing anything

Profiles: `docs/profile_index.txt`, `docs/profile_search.txt` (cProfile, sorted by self time),
`docs/flame_index.svg`, `docs/flame_search.svg` (py-spy, sampling).
Python vs native: cProfile lists C builtins (`{method 'translate' …}`, `{built-in method zlib…}`)
separately from Python functions, which is the split used below. Scalene 2.4 on Windows runs the
target inside its own thread and attributed all samples to `threading.py`, so it gave no usable
line-level attribution on this machine.

## `findex index` (serial build, 27.5 s under cProfile, 16 s without)

| # | Hotspot | Self time | Python / native | Hypothesis |
|---|---|---|---|---|
| 1 | `tokenize()` generator body | 6.3 s | Python | One generator resume per token (5.9 M) — interpreter overhead per token, not regex work. |
| 2 | `str.translate` per token | 4.2 s (5.9 M calls) | native, but call-bound | Each call is cheap; the cost is 5.9 M Python→C calls. Translating the *whole document once* before `finditer` should remove nearly all of it. |
| 3 | `build_index` loop + `PostingList.append` | 5.2 s + 4.1 s | Python | Per-(doc, term) Python work: Counter iteration, bounds check, three `array.append`s. Needs a different data flow (e.g. batch per document) to shrink. |

## `findex search` (Lab 3 engine, 3-term OR query, 50 runs = 0.34 s)

| # | Hotspot | Share | Python / native | Hypothesis |
|---|---|---|---|---|
| 1 | `BM25.score()` per posting (43 K calls) | ~60% | Python | Pure interpreter overhead: a method call, `df()`, `idf()`, `tf_weight()` per posting. Vectorizing over whole posting lists (NumPy) should give 10–100× on common terms. |
| 2 | dict accumulation in `search()` | ~15% | Python | Same cause; a dense `float64[N]` accumulator removes it. |
| 3 | `union_merge` | ~9% | Python | Two-pointer loop in Python; only matters for OR of large lists. Left alone (the scorer dominates). |

## Outcomes (after)

* Index #2 — **hypothesis rejected.** Translating apostrophes once per document instead of per
  token, measured directly (400 documents, best of 5): per-token 0.801 s, per-document 0.890 s —
  *slower*, because it copies the whole document. cProfile's own per-call overhead had made 5.9 M
  cheap C calls look like 4.2 s. Full serial builds were within noise (median 20.0 s before,
  18.1 s after, ranges overlapping). Reverted. Lesson: a deterministic profiler distorts
  call-heavy code; confirm with a direct benchmark before believing a hotspot.
* Search #1/#2: `fast.py` NumPy scorer — see the speedup table in README Lab 8.
