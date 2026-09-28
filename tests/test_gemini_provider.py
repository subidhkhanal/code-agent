"""Gemini adapter against a mocked HTTP transport (no key, no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from code_agent.llm.providers import GeminiProvider
from code_agent.llm.types import (
    CancelToken,
    Done,
    Message,
    ProviderError,
    Request,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    Usage,
)

KEY = "AIza-test-key-not-real-0123456789"


def sse(*chunks: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(c).encode() + b"\r\n\r\n" for c in chunks)


def provider(handler) -> GeminiProvider:
    return GeminiProvider(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)


READ_FILE = ToolSpec(
    "read_file",
    "Read a file",
    {"type": "object", "additionalProperties": False, "$schema": "x",
     "properties": {"path": {"type": "string", "description": "workspace path"}},
     "required": ["path"]},
)  # fmt: skip


def test_request_body_and_streamed_events():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        body = sse(
            {"candidates": [{"content": {"role": "model", "parts": [
                {"text": "thinking...", "thought": True}, {"text": "Let me "}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": [
                {"text": "read it."},
                {"functionCall": {"name": "read_file", "args": {"path": "a.py"}},
                 "thoughtSignature": "sig123"}]},
                "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30}},
        )  # fmt: skip
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    messages = (
        Message("system", "You are a coding agent."),
        Message("user", "fix it"),
        Message("assistant", "", (ToolCall("c0", "search_codebase", {"query": "x"}),)),
        Message("tool", "result 1", tool_call_id="c0", tool_name="search_codebase"),
        Message("tool", "result 2", tool_call_id="c0b", tool_name="search_codebase"),
    )
    events = list(provider(handler).stream(
        Request("some-model", messages, (READ_FILE,), 1000, 0.0), CancelToken()
    ))  # fmt: skip

    assert captured["url"].startswith(
        "https://generativelanguage.googleapis.com/v1beta/models/some-model:streamGenerateContent"
    )
    assert "alt=sse" in captured["url"] and KEY not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == KEY
    body = captured["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "You are a coding agent."}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert len(body["contents"][2]["parts"]) == 2  # consecutive tool results grouped
    assert body["contents"][2]["parts"][0]["functionResponse"]["name"] == "search_codebase"
    params = body["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "additionalProperties" not in params and "$schema" not in params
    assert body["generationConfig"] == {"maxOutputTokens": 1000, "temperature": 0.0}

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Let me read it."
    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert calls[0].name == "read_file" and calls[0].arguments == {"path": "a.py"}
    done = events[-1]
    assert isinstance(done, Done) and done.stop_reason == "tool_calls"
    assert done.usage == Usage(120, 30)
    assert done.provider_state is not None  # raw parts kept for the next turn
    assert done.provider_state["parts"][-1]["thoughtSignature"] == "sig123"


def test_provider_state_is_echoed_back_verbatim():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=sse({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
        )

    raw = [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "keep-me"}]
    msgs = (Message("user", "q"), Message("assistant", "", provider_state={"parts": raw}),
            Message("tool", "r", tool_call_id="1", tool_name="f"))  # fmt: skip
    list(provider(handler).stream(Request("m", msgs), CancelToken()))
    assert captured["body"]["contents"][1] == {"role": "model", "parts": raw}


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (503, True), (400, False), (403, False)]
)
def test_http_errors_are_classified(status: int, retryable: bool):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(ProviderError) as exc:
        list(provider(handler).stream(Request("m", (Message("user", "q"),)), CancelToken()))
    assert exc.value.retryable is retryable and exc.value.status == status
    assert KEY not in str(exc.value)


def test_transport_errors_are_retryable():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(ProviderError) as exc:
        list(provider(handler).stream(Request("m", (Message("user", "q"),)), CancelToken()))
    assert exc.value.retryable


def test_missing_key_is_not_retryable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ProviderError) as exc:
        list(provider(lambda r: httpx.Response(200)).stream(Request("m", ()), CancelToken()))
    assert not exc.value.retryable and "GEMINI_API_KEY" in str(exc.value)


def test_list_models_paginates_and_filters_to_generation_models():
    pages = {
        None: {"models": [
            {"name": "models/gen-a", "supportedGenerationMethods": ["generateContent"],
             "inputTokenLimit": 1000, "outputTokenLimit": 100},
            {"name": "models/embed-x", "supportedGenerationMethods": ["embedContent"]}],
            "nextPageToken": "p2"},
        "p2": {"models": [
            {"name": "models/gen-b", "supportedGenerationMethods": ["generateContent"]}]},
    }  # fmt: skip

    def handler(request):
        return httpx.Response(200, json=pages[request.url.params.get("pageToken")])

    models = provider(handler).list_models()
    assert [m.name for m in models] == ["gen-a", "gen-b"]
    assert models[0].input_token_limit == 1000 and models[1].input_token_limit is None


def test_cancel_stops_reading_the_stream():
    def handler(request):
        return httpx.Response(
            200, content=sse(*[{"candidates": [{"content": {"parts": [{"text": "x"}]}}]}] * 50)
        )

    cancel = CancelToken()
    stream = provider(handler).stream(Request("m", (Message("user", "q"),)), cancel)
    next(stream)
    cancel.cancel()
    with pytest.raises(Exception, match="cancelled"):
        list(stream)
