"""AWS Bedrock client (Converse API) with retries, concurrency and cost tracking."""

from __future__ import annotations

import logging
import time
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ConnectionClosedError, ReadTimeoutError

from core.types import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096

_PRICING: dict[str, tuple[float, float]] = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0),
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0),
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": (0.80, 4.0),
    "us.amazon.nova-lite-v1:0": (0.06, 0.24),
    "google.gemma-3-27b-it": (0.30, 0.30),
    "qwen.qwen3-32b-v1:0": (0.075, 0.300),
    "deepseek.v3.2": (0.62, 1.85),
    "us.deepseek.v3.2": (0.62, 1.85),
}

_RETRYABLE = ("ThrottlingException", "ServiceUnavailableException", "TooManyRequestsException")


class BedrockLLM:
    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        max_retries: int = 5,
        max_workers: int = 5,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self._max_retries = max_retries
        self._max_workers = max_workers

        boto_config = BotoConfig(
            region_name=region,
            retries={"max_attempts": max_retries, "mode": "adaptive"},
            max_pool_connections=max(max_workers, 1) + 5,
            read_timeout=300,
            connect_timeout=30,
        )
        self._client = boto3.client("bedrock-runtime", config=boto_config)

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

    def invoke(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.converse(**kwargs)
                break
            except ClientError as e:
                if e.response["Error"]["Code"] not in _RETRYABLE:
                    raise
                last_err = e
                time.sleep(min(2**attempt + 0.5, 30))
            except (ReadTimeoutError, ConnectionClosedError) as e:
                last_err = e
                time.sleep(min(2**attempt + 0.5, 30))
        else:
            raise last_err

        content = response["output"]["message"].get("content", [])
        text = content[0]["text"] if content and "text" in content[0] else ""

        usage = response["usage"]
        token_usage = TokenUsage(
            input_tokens=usage["inputTokens"],
            output_tokens=usage["outputTokens"],
            model_id=self.model_id,
        )
        self._total_input_tokens += token_usage.input_tokens
        self._total_output_tokens += token_usage.output_tokens
        self._total_calls += 1

        return LLMResponse(text=text, usage=token_usage, stop_reason=response.get("stopReason"))

    def invoke_batch(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_progress: Any | None = None,
    ) -> list[LLMResponse]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[LLMResponse | None] = [None] * len(message_batches)
        done = 0
        with ThreadPoolExecutor(max_workers=max(self._max_workers, 1)) as pool:
            futures = {
                pool.submit(
                    self.invoke, msgs, system=system,
                    temperature=temperature, max_tokens=max_tokens,
                ): i
                for i, msgs in enumerate(message_batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                done += 1
                if on_progress:
                    on_progress(done, len(message_batches))
        return results

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def estimated_cost_usd(self) -> float:
        input_price, output_price = _PRICING.get(self.model_id, (0.0, 0.0))
        return (
            self._total_input_tokens * input_price / 1_000_000
            + self._total_output_tokens * output_price / 1_000_000
        )

    def usage_summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }

    @staticmethod
    def user_message(text: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"text": text}]}
