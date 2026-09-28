"""Provider-neutral request/response types.

The agent loop only ever sees these. Each provider adapter translates to and from its own wire
format, so adding a provider never touches the loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # parsed JSON, *untrusted*: validated by the tool layer


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # role == "tool": which call this answers
    tool_name: str | None = None  # role == "tool": needed by providers that key on name
    # Opaque provider data that must be echoed back verbatim on later turns (e.g. Gemini's
    # thought signatures). Never inspected outside the adapter that produced it.
    provider_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (object); keep to the widely supported subset


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    call: ToolCall


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class Done:
    stop_reason: str  # provider's reason, e.g. "STOP", "MAX_TOKENS", "tool_calls"
    usage: Usage
    provider_state: dict[str, Any] | None = None  # to attach to the assistant Message


StreamEvent = TextDelta | ToolCallEvent | Done


@dataclass(frozen=True)
class Request:
    model: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    max_output_tokens: int = 4096
    temperature: float = 0.0


class ProviderError(RuntimeError):
    """A provider call failed. `retryable` decides between backoff-and-retry and giving up."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class CancelledError(RuntimeError):
    """The user cancelled (Ctrl+C) while a call was in flight."""


@dataclass
class CancelToken:
    """Shared flag checked between stream events, tool calls and retries."""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("cancelled by user")

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`; returns True early if cancelled."""
        return self._event.wait(seconds)
