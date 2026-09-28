"""Provider adapters: a scripted fake (tests, CI, demos) and Google Gemini (REST + SSE)."""

from __future__ import annotations

import json
import os
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

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

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ModelInfo:
    name: str
    input_token_limit: int | None = None  # context window, as reported by the provider
    output_token_limit: int | None = None


class Provider(Protocol):
    name: str

    def stream(self, request: Request, cancel: CancelToken) -> Iterator[StreamEvent]: ...

    def list_models(self) -> list[ModelInfo]: ...


# -- fake ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=lambda: Usage(1000, 200))
    error: ProviderError | None = None  # raise this instead of answering


FakeScriptItem = FakeTurn | Callable[[Request], FakeTurn]


@dataclass
class FakeProvider:
    """Deterministic provider. Each call consumes the next scripted turn; a callable turn
    receives the request, so a script can react to what the agent actually sent.

    Every request is recorded in `requests`: tests use this to assert on exactly what would have
    left the machine (e.g. that no secret was in any outbound payload)."""

    script: Iterable[FakeScriptItem] = ()
    models: tuple[ModelInfo, ...] = (
        ModelInfo("fake-cheap", 32_000, 4_096),
        ModelInfo("fake-strong", 128_000, 8_192),
    )
    chunk_size: int = 7  # stream text in small fragments to exercise the streaming parser
    name: str = "fake"
    requests: list[Request] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queue: deque[FakeScriptItem] = deque(self.script)

    def add(self, *items: FakeScriptItem) -> None:
        self._queue.extend(items)

    def stream(self, request: Request, cancel: CancelToken) -> Iterator[StreamEvent]:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("FakeProvider script exhausted: the agent made an extra call")
        item = self._queue.popleft()
        turn = item(request) if callable(item) else item
        if turn.error is not None:
            raise turn.error
        for i in range(0, len(turn.text), self.chunk_size):
            cancel.raise_if_cancelled()
            yield TextDelta(turn.text[i : i + self.chunk_size])
        for call in turn.tool_calls:
            yield ToolCallEvent(call)
        yield Done("tool_calls" if turn.tool_calls else "STOP", turn.usage)

    def list_models(self) -> list[ModelInfo]:
        return list(self.models)


# -- Gemini -------------------------------------------------------------------------------------

# Gemini accepts an OpenAPI-style subset of JSON Schema for function parameters.
_SCHEMA_KEYS = frozenset({"type", "description", "properties", "required", "items", "enum",
                          "nullable", "format", "minimum", "maximum"})  # fmt: skip


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties":
            out[key] = {name: _gemini_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _gemini_schema(value)
        else:
            out[key] = value
    return out


def _gemini_tools(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [{"functionDeclarations": [
        {"name": t.name, "description": t.description, "parameters": _gemini_schema(t.parameters)}
        for t in tools
    ]}]  # fmt: skip


def _gemini_contents(messages: tuple[Message, ...]) -> tuple[dict | None, list[dict]]:
    system_parts = [{"text": m.content} for m in messages if m.role == "system"]
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "assistant":
            if m.provider_state and "parts" in m.provider_state:
                parts = m.provider_state["parts"]  # verbatim, keeps thought signatures intact
            else:
                parts = ([{"text": m.content}] if m.content else []) + [
                    {"functionCall": {"name": c.name, "args": c.arguments}} for c in m.tool_calls
                ]
            contents.append({"role": "model", "parts": parts})
        elif m.role == "tool":
            part = {"functionResponse": {"name": m.tool_name, "response": {"result": m.content}}}
            # Consecutive tool results belong in one user turn.
            if contents and contents[-1]["role"] == "user" and contents[-1].get("_tool"):
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part], "_tool": True})
        else:
            contents.append({"role": "user", "parts": [{"text": m.content}]})
    for c in contents:
        c.pop("_tool", None)
    system = {"parts": system_parts} if system_parts else None
    return system, contents


@dataclass
class GeminiProvider:
    """Google Gemini via the public REST API (`streamGenerateContent?alt=sse`).

    The API key is read from the environment at call time and sent only in a header; it is never
    logged or stored."""

    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 120.0
    name: str = "gemini"
    transport: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"{self.api_key_env} is not set", retryable=False)
        return {"x-goog-api-key": key, "Content-Type": "application/json"}

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout_s, transport=self.transport)

    def list_models(self) -> list[ModelInfo]:
        found: list[ModelInfo] = []
        page: str | None = None
        with self._client() as client:
            while True:
                params = {"pageSize": "1000", **({"pageToken": page} if page else {})}
                resp = client.get(f"{self.base_url}/models", headers=self._headers(), params=params)
                data = self._checked(resp).json()
                found += [
                    ModelInfo(
                        m["name"].removeprefix("models/"),
                        m.get("inputTokenLimit"),
                        m.get("outputTokenLimit"),
                    )
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                page = data.get("nextPageToken")
                if not page:
                    return found

    @staticmethod
    def _checked(resp: httpx.Response) -> httpx.Response:
        if resp.status_code >= 400:
            resp.read()
            detail = resp.text[:300]
            raise ProviderError(
                f"gemini HTTP {resp.status_code}: {detail}",
                retryable=resp.status_code in RETRYABLE_STATUS,
                status=resp.status_code,
            )
        return resp

    def stream(self, request: Request, cancel: CancelToken) -> Iterator[StreamEvent]:
        system, contents = _gemini_contents(request.messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_output_tokens,
                "temperature": request.temperature,
            },
        }
        if system:
            body["systemInstruction"] = system
        if request.tools:
            body["tools"] = _gemini_tools(request.tools)
        url = f"{self.base_url}/models/{request.model}:streamGenerateContent"

        usage = Usage()
        finish = "STOP"
        model_parts: list[dict[str, Any]] = []
        saw_call = False
        try:
            with (
                self._client() as client,
                client.stream(
                    "POST", url, params={"alt": "sse"}, headers=self._headers(), json=body
                ) as resp,
            ):
                self._checked(resp)
                for line in resp.iter_lines():
                    cancel.raise_if_cancelled()
                    if not line.startswith("data:"):
                        continue
                    chunk = json.loads(line[5:])
                    meta = chunk.get("usageMetadata") or {}
                    if meta:
                        usage = Usage(meta.get("promptTokenCount", 0),
                                      meta.get("candidatesTokenCount", 0))  # fmt: skip
                    for cand in chunk.get("candidates", [])[:1]:
                        finish = cand.get("finishReason", finish)
                        for part in (cand.get("content") or {}).get("parts", []):
                            model_parts.append(part)
                            if "text" in part and not part.get("thought"):
                                yield TextDelta(part["text"])
                            elif "functionCall" in part:
                                saw_call = True
                                fc = part["functionCall"]
                                yield ToolCallEvent(ToolCall(
                                    id=fc.get("id") or uuid.uuid4().hex[:12],
                                    name=fc.get("name", ""),
                                    arguments=dict(fc.get("args") or {}),
                                ))  # fmt: skip
        except httpx.TimeoutException as exc:
            raise ProviderError(f"gemini timeout: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"gemini connection error: {exc}", retryable=True) from exc
        yield Done("tool_calls" if saw_call else finish, usage, {"parts": model_parts})
