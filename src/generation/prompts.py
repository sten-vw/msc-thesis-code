"""Shared system prompt for the strategies that generate a fresh doc->query pair.

InPars, Promptagator, DUQGen, and UDAPDR share one persona so differences
in query character reflect the generation algorithm, not prompt style.
Naive generation omits it; DRAGON's stages use their own system prompts.
"""

from __future__ import annotations

QUERY_GEN_SYSTEM = (
    "You are a search query generator. Given a document, write a "
    "question that the document can answer. The question should be "
    "natural — what a real user might type into a search engine. "
    "Output ONLY the question text, with no labels, prefixes, "
    "numbering, markdown headers, bullet points, quotation marks, or "
    "any explanation. Do not echo any section headers from the "
    "examples."
)
