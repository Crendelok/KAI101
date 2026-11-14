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
