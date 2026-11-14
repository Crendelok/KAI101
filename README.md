# findex

A full-text search engine built from scratch, one lab at a time.
**Lab 01:** a corpus loader and tokenizer that stream the corpus instead of loading it.

## Corpus

Public-domain books from [Project Gutenberg](https://www.gutenberg.org/), one book = one document,
licence header/footer stripped. `data/` is git-ignored; to get the corpus:

```bash
uv run python scripts/fetch_gutenberg.py          # ~40 popular books into data/gutenberg/
uv run python scripts/fetch_gutenberg.py 1342 84  # or pick ebook IDs
```

`iter_documents` also reads directories of `.txt`, JSON-lines files (`{"text": ...}` per line,
e.g. arXiv abstracts), and `.gz` / `.bz2` versions of both without extracting them.

## Usage

```bash
uv sync
uv run python -m findex.stats data/                # doc count, tokens, vocab, top-50, time, peak mem
uv run python -m findex.stats data/ --limit 10     # first 10 documents only (itertools.islice)
uv run python -m findex.stats data/ --compare      # eager vs lazy, as a markdown table
uv run pytest
uv run ruff check
uv run python scripts/repl_experiments.py          # the five "prove it to yourself" experiments
```

## Layout

| File | What |
|---|---|
| `src/findex/corpus.py` | `iter_documents(root) -> Iterator[Document]` — generator, one file open at a time |
| `src/findex/tokenize.py` | `tokenize(text) -> Iterator[str]` — NFC → casefold → `re.finditer` |
| `src/findex/stats.py` | the lazy pipeline, the eager twin, `tracemalloc` measurement, CLI |
| `tests/` | tokenizer policy checks + corpus loader checks |

## Tokenizer policy

| Case | Input | Tokens |
|---|---|---|
| case | `Straße STRASSE` | `strasse`, `strasse` (`casefold`, not `lower`) |
| accents | `café` vs `cafe` + U+0301 | both `café` (NFC) |
| apostrophe inside a word | `don't`, `don’t`, `п'ять`, `пʼять` | kept, unified to `'` |
| quotes around a word | `'quoted'` | `quoted` |
| hyphen | `state-of-the-art` | `state`, `of`, `the`, `art` |
| digits | `Python3 2008 3.14` | `python3`, `2008`, `3`, `14` |
| underscore | `snake_case` | `snake`, `case` |

Letters are any Unicode letter/digit (`\w` minus `_`), so Cyrillic, Greek and CJK work.
Combining marks stay attached to their letter when NFC has no precomposed form.

## Eager vs lazy

Measured with `python -m findex.stats data/ --compare` (same corpus, same tokenizer, `tracemalloc`
peak; tracemalloc slows both runs equally).

| Version | Documents | Tokens | Peak memory | Elapsed |
|---|---|---|---|---|
| eager (lists) | 40 | 5,855,520 | 406.3 MiB | 44.83 s |
| lazy (generators) | 40 | 5,855,520 | 54.9 MiB | 40.63 s |

Corpus: 40 Gutenberg books, 32.6 MB on disk, 76,552 distinct terms. Windows 11, CPython 3.12.10.
Both versions produce the identical `Counter` (the `--compare` run asserts it).

The eager version keeps everything alive at once: all 40 decoded texts (a 32 MB corpus becomes
~60 MB of `str`, because a single curly quote forces CPython to store a whole book at 2 bytes per
character), then a list of lists holding 5.9 million token references, then a second flat list of
the same 5.9 million references, and only then the `Counter`. Each list slot is an 8-byte pointer
plus a separate string object per token, which is why it reaches ~400 MiB: over 10× the corpus
size. The lazy version never holds more than one book. Its peak is set by the largest book
(Les Misérables, 3.3 MB): the raw bytes, the decoded `str` (6.4 MiB), the NFC and casefolded
copies, and `unicodedata`'s temporary buffer, together with the `Counter` (5.5 MiB, still in
memory after the run because it *is* the answer). Memory isn't zero because a document is the
unit of streaming. To go lower, tokenize line by line instead of book by book. The lazy run is
also slightly faster: it never allocates or frees millions of list slots.

> After Lab 1 one book (ebook 30254) was swapped for Plato's *Republic* (1497), and from Lab 2 on
> the books are split into ~1,500-word sections (`scripts/make_sections.py`): **3,876 documents**
> in `data/sections/`, a corpus where ranking has something to rank.

## Lab 02 — the inverted index

```bash
uv run python scripts/make_sections.py
uv run python -m findex.index data/sections --out index.fidx --positions
uv run python -m findex.search index.fidx "whale ship NOT boat" --boolean --engine merge
uv run python scripts/bench_lab02.py data/sections      # all three tables below
```

* `index.py` — one streaming pass: each document's terms are counted in a per-document `Counter`
  (or position lists with `--positions`) and flushed into `defaultdict(PostingList)` when the
  document ends, so postings come out sorted by doc id for free. `Posting` and `DocMeta` are
  `@dataclass(frozen=True, slots=True)`, hashable values.
* `setops.py` — two-pointer merge for AND / OR / AND-NOT, and a set-based engine behind the same
  interface (`--engine merge|set`).
* `store.py` — `pickle` and a binary `.fidx` format: postings as raw `uint32` arrays, a JSON header
  mapping term → byte offset, zlib-compressed document texts. `.fidx` can be loaded eagerly or
  `mmap`ed lazily, so load time stops depending on index size. It holds only data, so it's safe to load
  a downloaded file. **`pickle` is not**: unpickling runs whatever callable the file names
  (`__reduce__`), so loading a downloaded `.pkl` is remote code execution. Only load pickles you wrote.

### Postings representation (3,876 docs, 5.8 M tokens, 76,766 terms, no positions)

| Postings representation | Peak memory (build) | Index file size (pickle) | Load time |
|---|---|---|---|
| `list[Posting]` with plain `@dataclass` | 204.0 MiB | 41.7 MiB | 1.85 s |
| `list[Posting]` with `slots=True` | 135.3 MiB | 33.1 MiB | 7.08 s |
| `array('I')` pairs (doc_ids, tfs) — no objects | **43.4 MiB** | **20.5 MiB** | **0.21 s** |

There are ~1.2 M postings. As a plain dataclass, each posting is a full object: 16 bytes of header,
a pointer to its own `__dict__`, and the dict itself (~100+ bytes for two keys), plus the 8-byte
slot in the list that points at it. That's roughly 150 bytes per posting. `slots=True` deletes the
`__dict__`: the instance becomes a header plus two inline pointers (48 bytes). The ints themselves
are mostly small-int singletons, so they cost nothing extra. With `array('I')` there are no objects
at all: one posting is two 4-byte machine integers in a contiguous buffer, ~8 bytes. The
remaining ~40 MiB is the dict of 77 K term strings and per-array overhead. The pickle numbers say
the same thing, plus one surprise: slotted *frozen* instances load 4× slower than plain ones. A
frozen slotted class has no `__dict__` to fill, so unpickling calls `__setstate__`, which goes
through `object.__setattr__` once per field per object. An array unpickles as a single `bytes`
copy. **Production `findex` therefore stores arrays** (`PostingList`) and creates `Posting`
objects only on demand.

### Two-pointer merge vs `set` (µs per operation)

| Query | df | merge | set | Faster |
|---|---|---|---|---|
| `and` AND `to` (two most common) | 3,876 / 3,876 | 861 | 348 | set 2.5× |
| `and` OR `to` | 3,876 / 3,876 | 895 | 348 | set 2.6× |
| `publick` AND `1729` (two rarest, df ≥ 2) | 2 / 2 | 1 | 1 | tie |
| `publick` OR `1729` | 2 / 2 | 1 | 1 | tie |
| `to` AND `tumbling` (common × mid) | 3,876 / 50 | 581 | 114 | set 5.1× |

On paper the merge wins: O(n + m) with no hashing and no final sort. In CPython the set engine
still wins on the big lists because `set(a) & set(b)` runs entirely in C, while the merge pays
interpreter overhead (bytecode dispatch, boxing ints out of the array) on every step. On tiny lists
the two are equal because call overhead dominates. The merge's real advantages show up later: it
streams, keeps order without a sort, and vectorizes (Lab 8). `merge` stays the default for those
reasons, and `--engine set` is there for comparison.

### Storage formats

| Index | Format | File size | Save | Load |
|---|---|---|---|---|
| no positions | pickle | 37.1 MiB | 0.70 s | 341 ms |
| | `.fidx` eager | 34.7 MiB | 0.62 s | 473 ms |
| | `.fidx` mmap (lazy) | 34.7 MiB | 0.57 s | **141 ms** |
| with positions | pickle | 60.8 MiB | 0.91 s | 661 ms |
| | `.fidx` eager | 57.3 MiB | 0.53 s | 618 ms |
| | `.fidx` mmap (lazy) | 57.3 MiB | 0.72 s | **155 ms** |

Sizes include ~19 MiB of compressed document text (needed for snippets and the document view).
Positions add ~23 MiB, about 2.3× the postings alone, and build peak memory goes from 60 to 90 MiB.
The positions flag stays on: phrase queries and snippets depend on it. The lazy load costs
about the same whether or not positions are stored, because it only parses the header and maps
the file.

## Lab 03 — ranking, the query language, and a real `Index`

```bash
uv run python -m findex.search index.fidx '"white whale" captain' --k 3 --repeat 2 -v
uv run python scripts/eval_lab03.py index.fidx
```

* **`Index` is a `collections.abc.Mapping[str, PostingList]`:** `len(ix)`, `"whale" in ix`, `ix["whale"]`
  (`KeyError` if absent), `for term in ix`, `repr(ix)`. Also `num_docs`, `avg_doc_length`
  (`cached_property`), `doc_length()`, `df()`, `text()`. `Mapping` brings an `__eq__`, which sets
  `__hash__ = None`, so `Index` restores identity equality and hashing: an index is an entity, not
  a value, and the query cache needs to hash it. `store.open_index()` is a `@contextmanager` that
  closes the mmap even when the `with` body raises (tested).
* **Ranking (`scoring.py`):** a `Scorer` protocol with `score(term, posting, index)`, implemented
  by `TfIdf` and `BM25(k1=1.5, b=0.75)`. `search()` adds up scores per document and takes the top k
  with `heapq.nlargest` (O(n log k), not O(n log n)). It returns a `ResultList`, a `Sequence` whose
  slices keep `.total`, made of `SearchResult` dataclasses with `order=True` that sort best-first.
* **Query language (`query.py`):** a recursive-descent parser builds `Term` / `Phrase` / `And` / `Or` / `Not`
  frozen dataclasses (which overload `&`, `|`, `~`). It handles implicit AND, `OR`, `NOT`,
  parentheses and `"quoted phrases"` (checked against positions). Tests compare trees, for example
  `parse('a OR b c') == Or(Term('a'), And(Term('b'), Term('c')))`. `a NOT b` is evaluated as a
  difference, never as "everything minus b".
* **Decorators and caching:** `@timed` (with `functools.wraps`) on `build_index`, `save`, `load`,
  `matching_ids`, `search`. `functools.lru_cache(256)` on query → matching ids. In the log below,
  a repeated query skips parsing and evaluation:

  ```text
  DEBUG findex.timing: matching_ids took 1.5 ms
  DEBUG findex.timing: search took 2.1 ms
  DEBUG __main__: query cache hit for '"white whale" captain'
  DEBUG findex.timing: matching_ids took 0.1 ms
  DEBUG findex.timing: search took 0.5 ms
  ```
* **Snippets (`snippets.py`):** a ±80-character window centred on the match that covers the most
  distinct query terms, snapped to word boundaries, with spans the caller renders as `«plain»`,
  rich markup, or `<mark>`-escaped HTML.

  ```text
  1.  15.472  Moby-Dick — part 43/141
      …“«Captain» Ahab,” said Tashtego, “that «white» «whale» must be the same that some call…
  ```

### Ranking sanity checks (BM25, real corpus)

1. **Rare beats common.** `harpoon` has df 40 and idf 4.56; `sea` has df 687 and idf 1.73. For
   `harpoon OR sea`, all 5 of the top 5 contain `harpoon`, even though hundreds of sections say
   `sea` more often.
2. **Saturation.** At average length, the BM25 tf weight for tf = 1, 2, 5, 10, 19, 20 is 1.000,
   1.429, 1.923, 2.174, 2.317, 2.326. The 20th repetition adds 0.0085, 0.9% of what the first one
   added. (TF-IDF's `1 + ln tf` would still add 0.05.)
3. **Short beats long.** `telescope` appearing once in a 1,175-token section scores 5.601; once in
   a 1,571-token section, 4.955. The same checks run as unit tests on the fixture corpus
   (`tests/test_search.py`).

### precision@5 — TF-IDF vs BM25

Ten queries written to avoid character names. A section counts as relevant if it comes from the
book the query is about (`eval/queries.json`). The words are OR-ed so ranking decides.

| Query | Relevant book | TF-IDF | BM25 |
|---|---|---|---|
| whale ship captain harpoon | Moby-Dick | 1.0 | 1.0 |
| detective crime clue | The Adventures of Sherlock Holmes | 0.6 | 0.8 |
| vampire blood garlic | Dracula | 0.8 | 0.8 |
| battle emperor army retreat | War and Peace | 0.8 | 0.8 |
| prince fortune cruelty state | The Prince | 1.0 | 1.0 |
| marriage fortune ball sisters | Pride and Prejudice | 0.8 | 1.0 |
| pirates treasure island map | Treasure Island | 1.0 | 0.8 |
| creature creation science despair | Frankenstein | 0.8 | 0.8 |
| sovereign commonwealth covenant | Leviathan | 1.0 | 1.0 |
| justice soul guardians city | The Republic | 1.0 | 1.0 |
| **mean** | | **0.88** | **0.90** |

The sample is small and noisy, and both scorers are good when the words are distinctive. The
differences come from saturation. For *detective crime clue*, TF-IDF lets a *Tom Sawyer* section
into the top 5 on the strength of repeating one word; under BM25 extra repetitions are worth
little, so a fourth Holmes story takes that place. Saturation also explains BM25's one loss: on
*pirates treasure island map*, Treasure Island's sections no longer get much credit for saying
"island" and "treasure" over and over, so an *Odyssey* section about islands reaches #2. Sections
are all about 1,500 words, so length normalization barely matters here.

## Lab 04 — a real tool: typed, tested, packaged

<!-- After pushing to GitHub, replace OWNER/REPO: -->
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

```bash
uv tool install .                     # puts `findex` on PATH
findex index data/sections --out index.fidx
findex search index.fidx '"white whale" captain' -k 5
findex search index.fidx 'whale OR ship' --json | jq .title
findex stats index.fidx
findex -vv search index.fidx 'whale'  # timing + cache logs on stderr
```

From a release wheel, on any machine with `uv`:

```bash
uv build                                              # dist/findex-0.4.0-py3-none-any.whl
uvx --from ./dist/findex-0.4.0-py3-none-any.whl findex --help
uv tool install https://github.com/OWNER/REPO/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl
```

* **Typing:** `pyright` in strict mode, 0 errors. `Scorer` is a `typing.Protocol`. Generators are
  typed `Iterator[...]` and accept `Iterable[...]`; `Literal` is used for the engine, executor and
  scorer names; `@timed` uses PEP 695 generics (`def timed[**P, R]`) so decorated functions keep
  their signatures. Each of the four `# pyright: ignore` comments has its reason next to it
  (untyped JSON values, and a typeshed generic on `TextIOWrapper`).
* **Tests:** 80 pytest tests. A session-scoped fixture corpus lives in `conftest.py`. The tokenizer
  tests are parametrized and include Unicode. The query parser is tested by tree equality, the
  three Lab 3 ranking checks run as tests, save/load round-trips through both formats and the mmap,
  and the CLI runs through `typer.testing.CliRunner`. **Five Hypothesis properties:** merge equals
  set algebra; tokens are never empty and never contain whitespace; the tokenizer is
  case-insensitive and stable on ASCII text; any query tree survives `parse(str(tree))`; and any
  small corpus keeps its index invariants and survives a save/load round trip.
* **Coverage: 81%** of the package (branch coverage). What isn't covered, and why: the
  `python -m findex.index/search` argparse mains, which are thin Lab 2 entry points now superseded
  by the CLI; the decode-error and unreadable-archive branches in `corpus.py`; and `parallel.py`,
  which is Lab 5's code arriving early. Excluding `parallel.py`, statement coverage is 89%.
* **CLI:** a `typer` app with `index` (a `rich` progress bar on stderr), `search` (a `rich` table
  with highlighted snippets, or `--json` lines on stdout) and `stats`. `-v`/`-vv` set the log level.
  Logging is configured once in `cli.py` and every diagnostic goes to stderr. Expected failures print
  one line and exit nonzero: a missing file exits 1, a bad query exits 2.
* **Packaging:** PEP 621 `pyproject.toml`, the version read from `findex/__init__.py`, a `dev`
  dependency group, `[project.scripts] findex = "findex.cli:app"`, `py.typed`, and a committed
  `uv.lock`. `uv build` produces the wheel above. It was verified in a clean, isolated environment
  with `uvx --from dist/…whl findex --help`.
* **CI:** `.github/workflows/ci.yml` runs ruff, `ruff format --check`, pyright and pytest with
  coverage on Python 3.12 and 3.13, then builds the wheel. Pushing a `v*` tag attaches the wheel to
  a GitHub Release.

> **Windows note:** if the project lives under a very long path, files deep inside `.venv` can
> exceed the 260-character `MAX_PATH` limit and fail to import. The symptom is
> `ImportError … (unknown location)`. Point uv at a shorter venv path with
> `UV_PROJECT_ENVIRONMENT=C:\venvs\findex`, or enable long paths in Windows.
