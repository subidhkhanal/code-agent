"""Anthropic Claude via the official `anthropic` SDK (Messages API, streaming).

Translation rules (see docs/adr/0011):
* System messages become the top-level `system`; tool results become `tool_result` blocks, all
  results of one turn in a single user message, as the API requires.
* The assistant's raw content blocks (thinking blocks with their signatures included) are kept
  in `provider_state` and replayed verbatim on later turns. Opus-class models must see their own
  thinking blocks unchanged to continue a tool-use conversation.
* The loop elides old tool output to stay within budget, which changes the conversation prefix a
  thinking block was produced under. `drop_mismatched_thinking` asks the API to drop such blocks
  instead of rejecting the request.
* `refusal_fallback` opts into the server-side refusal fallback: a safety decline is re-run on
  Anthropic's recommended fallback model inside the same call. A final `refusal` stop reason
  still surfaces, and the loop reports it.
* Prompt caching is on (top-level `cache_control`): every turn of an agent task resends the same
  system prompt, tools and retrieved code, which then bill at the cache-read rate.
* No `temperature`: current Claude models reject non-default sampling parameters.
* The SDK's own retries are off; the gateway owns retries and fallbacks so budgets and backoff
  behave the same for every provider.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import anthropic

from code_agent.llm.providers import RETRYABLE_STATUS, ModelInfo
from code_agent.llm.types import (
    CancelToken,
    Done,
    Message,
    ProviderError,
    Request,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    Usage,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"
OVERLOADED = 529  # Anthropic's "overloaded" status: transient, worth a retry

# Block types that are only meaningful to the model that produced them. After a mid-output
# refusal fallback they must not be echoed from before the fallback boundary.
_MODEL_INTERNAL = frozenset({"thinking", "redacted_thinking", "tool_use"})


def _claude_tools(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    # Tool inputs here are small (paths, queries, one command); edits stream as text, not tool
    # input. So eager input streaming would buy nothing, and leaving it off keeps the API's own
    # validation of tool inputs against the schema.
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
    ]


def _replayable(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Content blocks as they must be echoed back. After a mid-output refusal fallback, model-
    internal blocks before the last `fallback` marker are dropped (the marker itself is an audit
    note the API ignores, so it goes too)."""
    last = max((i for i, b in enumerate(content) if b.get("type") == "fallback"), default=-1)
    if last < 0:
        return content
    before = [b for b in content[:last] if b.get("type") not in _MODEL_INTERNAL | {"fallback"}]
    return before + content[last + 1 :]


def _claude_messages(messages: tuple[Message, ...]) -> tuple[list[dict], list[dict]]:
    system = [{"type": "text", "text": m.content} for m in messages if m.role == "system"]
    out: list[dict[str, Any]] = []

    def user_blocks() -> list[dict[str, Any]]:
        # Consecutive user-side messages (tool results, then a note) share one user turn, with
        # tool results first.
        if out and out[-1]["role"] == "user":
            return out[-1]["content"]
        out.append({"role": "user", "content": []})
        return out[-1]["content"]

    for m in messages:
        if m.role == "system":
            continue
        if m.role == "assistant":
            if m.provider_state and "content" in m.provider_state:
                blocks = list(m.provider_state["content"])
            else:
                blocks = ([{"type": "text", "text": m.content}] if m.content else []) + [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
            out.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            user_blocks().append(
                {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            )
        elif m.content:
            user_blocks().append({"type": "text", "text": m.content})
    return system, out


def _usage(raw: Any) -> Usage:
    read = getattr(raw, "cache_read_input_tokens", None) or 0
    write = getattr(raw, "cache_creation_input_tokens", None) or 0
    return Usage(
        input_tokens=(raw.input_tokens or 0) + read + write,
        output_tokens=raw.output_tokens or 0,
        cache_read_tokens=read,
        cache_write_tokens=write,
    )


def _error(exc: Exception) -> ProviderError:
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        return ProviderError(
            f"anthropic HTTP {status}: {exc.message[:300]}",
            retryable=status in RETRYABLE_STATUS or status == OVERLOADED,
            status=status,
        )
    if isinstance(exc, anthropic.APITimeoutError):
        return ProviderError(f"anthropic timeout: {exc}", retryable=True)
    return ProviderError(f"anthropic connection error: {exc}", retryable=True)


@dataclass
class AnthropicProvider:
    """The key comes from `api_key` if given (the playground holds it in memory only), else from
    the environment at call time. It is sent only by the SDK, in a header, and never logged."""

    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    timeout_s: float = 120.0
    effort: str | None = None
    refusal_fallback: bool = True
    drop_mismatched_thinking: bool = True
    name: str = "anthropic"
    http_client: Any = None  # tests inject an anthropic.DefaultHttpxClient with a mock transport

    def _client(self) -> anthropic.Anthropic:
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"{self.api_key_env} is not set", retryable=False)
        return anthropic.Anthropic(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            max_retries=0,
            http_client=self.http_client,
        )

    def list_models(self) -> list[ModelInfo]:
        try:
            return [
                ModelInfo(m.id, m.max_input_tokens, m.max_tokens)
                for m in self._client().models.list(limit=1000)
            ]
        except anthropic.APIError as exc:
            raise _error(exc) from exc

    def _params(self, request: Request) -> dict[str, Any]:
        system, messages = _claude_messages(request.messages)
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            params["system"] = system
        if request.tools:
            params["tools"] = _claude_tools(request.tools)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        betas: list[str] = []
        if self.refusal_fallback:
            betas.append(FALLBACK_BETA)
            params["fallbacks"] = "default"
        if self.drop_mismatched_thinking:
            betas.append(THINKING_BINDING_BETA)
            params["thinking"] = {
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
        if betas:
            params["betas"] = betas
        return params

    def stream(self, request: Request, cancel: CancelToken) -> Iterator[StreamEvent]:
        params = self._params(request)
        try:
            with self._client().beta.messages.stream(**params) as stream:
                for event in stream:
                    cancel.raise_if_cancelled()
                    if event.type == "text":
                        yield TextDelta(event.text)
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            raise _error(exc) from exc

        content = [block.model_dump(exclude_none=True) for block in final.content]
        stop = final.stop_reason or "end_turn"
        calls = [b for b in final.content if b.type == "tool_use"]
        # A refusal or a max_tokens cut can leave a tool_use block with incomplete input:
        # never hand that to the tool layer.
        if stop == "tool_use":
            for block in calls:
                args = block.input if isinstance(block.input, dict) else {}
                yield ToolCallEvent(ToolCall(block.id, block.name, dict(args)))
        else:
            content = [b for b in content if b.get("type") != "tool_use"]
        yield Done(
            "tool_calls" if stop == "tool_use" else stop,
            _usage(final.usage),
            {"content": _replayable(content)},
        )
