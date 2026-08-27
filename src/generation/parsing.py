"""Shared response parsing for generation strategies.

One canonical place for line-level cleanup: drops blank and "Bad Question"
lines, strips echoed labels ("Query:", "Relevant Query:", etc.) and
wrapping quotes. :func:`parse_queries` caps results for multi-query
strategies; :func:`first_query` returns the first line for single-query
strategies; :func:`clean_query_line` is the shared per-line helper.
"""

from __future__ import annotations

from core.types import Document, Query
from generation.base import make_query

_DEFAULT_PREFIXES: tuple[str, ...] = (
    "good question",
    "query",
    "relevant query",
    "question",
    "rewritten question",
)

_DROP_LINE_PREFIXES: tuple[str, ...] = ("bad question",)


def clean_query_line(
    line: str,
    *,
    strip_prefixes: tuple[str, ...] = _DEFAULT_PREFIXES,
    drop_prefixes: tuple[str, ...] = _DROP_LINE_PREFIXES,
) -> str:
    """Clean one line of model output; returns ``""`` if blank, drop-prefixed, or meta-only."""
    line = line.strip()
    if not line:
        return ""

    while line and line[0] in "#-*":
        line = line[1:].lstrip()
    if not line:
        return ""

    lowered = line.lower()
    for drop in drop_prefixes:
        if lowered.startswith(drop):
            return ""

    if ":" in line:
        head, _, rest = line.partition(":")
        if _looks_like_label(head, strip_prefixes):
            line = rest.strip()

    if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
        line = line[1:-1].strip()

    if line and _is_meta_only(line.lower(), strip_prefixes):
        return ""

    return line


_LABEL_KEYWORDS: frozenset[str] = frozenset({"query", "question"})


def _looks_like_label(head: str, strip_prefixes: tuple[str, ...]) -> bool:
    """True if ``head`` looks like a section label: matches a strip prefix, or is short (<=8
    tokens) with "query"/"question" as a token.
    """
    head_norm = head.strip().lower()
    if not head_norm:
        return False
    if any(head_norm.startswith(p) for p in strip_prefixes):
        return True
    tokens = [t.strip(".,;") for t in head_norm.split()]
    if len(tokens) > 8:
        return False
    return any(t in _LABEL_KEYWORDS for t in tokens)


def _is_meta_only(text: str, label_prefixes: tuple[str, ...]) -> bool:
    """True if ``text`` is only label/filler words (e.g. "search query"), with no real question
    content.
    """
    if any(ch in text for ch in "?!"):
        return False
    label_words: set[str] = set()
    for prefix in label_prefixes:
        label_words.update(prefix.split())
    label_words.update({"search", "generated", "synthetic", "example",
                        "for", "of", "the", "a", "an"})
    tokens = [t for t in text.replace(",", " ").split() if t]
    if not tokens:
        return True
    return all(t.strip(".:,;").isdigit() or t.strip(".:,;") in label_words
               for t in tokens)


def parse_queries(
    response_text: str,
    source_doc: Document,
    strategy: str,
    *,
    max_queries: int | None = None,
) -> list[Query]:
    """Parse every cleaned line in ``response_text`` into a Query, capped at ``max_queries``
    results.
    """
    queries: list[Query] = []
    for raw in response_text.strip().split("\n"):
        cleaned = clean_query_line(raw)
        if not cleaned:
            continue
        queries.append(make_query(cleaned, source_doc, strategy))
        if max_queries is not None and len(queries) >= max_queries:
            break
    return queries


def first_query(response_text: str) -> str:
    """Return the first cleaned, non-empty line of the response, or ``""`` if none."""
    for raw in response_text.strip().split("\n"):
        cleaned = clean_query_line(raw)
        if cleaned:
            return cleaned
    return ""
