"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    source_doc_id: str | None = None
    generation_strategy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelevanceJudgments:
    judgments: dict[str, dict[str, int]]

    def get(self, query_id: str) -> dict[str, int]:
        return self.judgments.get(query_id, {})

    def query_ids(self) -> set[str]:
        return set(self.judgments.keys())


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    model_id: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage
    stop_reason: str | None = None
