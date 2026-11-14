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
