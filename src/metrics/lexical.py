"""Lexical overlap between a query and its source passage.

Stopword-filtered, Porter-stemmed content tokens; coverage is the fraction of
the query's content terms that also occur in the passage.
"""

from __future__ import annotations

import re

_STOP = set(
    """a an and are as at be by for from has have he in is it its of on that the to was were will with
    this these those there their them they i you your we our us he she his her him my me do does did done
    or but if then else when while which who whom whose what where why how not no nor so than too very can
    could should would may might must shall about above after again against all am any because been before
    being below between both during each few further here into more most other own same some such only off
    over under up down out also into per via within without upon among across""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _porter():
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer().stem
    except Exception:
        def fallback(w):
            for suf in ("ing", "edly", "ed", "ly", "es", "s"):
                if len(w) > len(suf) + 2 and w.endswith(suf):
                    return w[: -len(suf)]
            return w

        return fallback


_STEM = _porter()


def content_set(text: str) -> set[str]:
    if not text:
        return set()
    out = set()
    for t in _WORD.findall(text.lower()):
        if t in _STOP or len(t) == 1:
            continue
        out.add(_STEM(t))
    return out


def coverage(cq: set[str], cd: set[str]) -> float:
    if not cq:
        return float("nan")
    return len(cq & cd) / len(cq)
