"""A steady search workload to attach a sampling profiler to.

py-spy record -o docs/flame_search.svg -- python scripts/search_loop.py python
"""

import sys

from findex.fast import search_fast
from findex.search import _matching_ids, search
from findex.store import load

engine = search_fast if (sys.argv[1:] or ["python"])[0] == "numpy" else search
index = load("index.fidx")
for _ in range(150):
    _matching_ids.cache_clear()  # profile the whole path, not a cache hit
    engine(index, "whale OR ship OR captain", k=10)
