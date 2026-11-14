"""Boolean operations on sorted doc-id lists (Lab 2).

Two engines with one interface:

* ``merge`` — the classic two-pointer walk over two sorted lists, O(n + m),
  no hashing, output already sorted.
* ``set``   — build Python sets and use ``&``/``|``/``-``, then sort the result.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import Literal

type Engine = Literal["merge", "set"]
type DocIds = Sequence[int]


def intersect_merge(a: DocIds, b: DocIds) -> array[int]:
    out = array("I")
    i = j = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def union_merge(a: DocIds, b: DocIds) -> array[int]:
    out = array("I")
    i = j = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference_merge(a: DocIds, b: DocIds) -> array[int]:
    """Elements of `a` not in `b`."""
    out = array("I")
    i = j = 0
    na, nb = len(a), len(b)
    while i < na:
        if j >= nb:
            out.extend(a[i:])
            break
        x, y = a[i], b[j]
        if x == y:
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            j += 1
    return out


def intersect_set(a: DocIds, b: DocIds) -> array[int]:
    return array("I", sorted(set(a) & set(b)))


def union_set(a: DocIds, b: DocIds) -> array[int]:
    return array("I", sorted(set(a) | set(b)))


def difference_set(a: DocIds, b: DocIds) -> array[int]:
    return array("I", sorted(set(a) - set(b)))


OPS = {
    "merge": (intersect_merge, union_merge, difference_merge),
    "set": (intersect_set, union_set, difference_set),
}


def intersect(a: DocIds, b: DocIds, engine: Engine = "merge") -> array[int]:
    return OPS[engine][0](a, b)


def union(a: DocIds, b: DocIds, engine: Engine = "merge") -> array[int]:
    return OPS[engine][1](a, b)


def difference(a: DocIds, b: DocIds, engine: Engine = "merge") -> array[int]:
    return OPS[engine][2](a, b)
