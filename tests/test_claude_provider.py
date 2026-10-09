"""AnthropicProvider against a mocked HTTP transport: wire format in, events out. No network."""

from __future__ import annotations

import json

import anthropic
import httpx2
import pytest

from code_agent.config import LLMConfig
from code_agent.llm.claude import (
    FALLBACK_BETA,
    THINKING_BINDING_BETA,
    AnthropicProvider,
    _replayable,
)
from code_agent.llm.factory import build_gateway
from code_agent.llm.gateway import Price
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

MODEL = "claude-test-model"  # any id: the provider never interprets model names


def sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_events(blocks: list[dict], stop: str, usage: dict | None = None) -> bytes:
    """A full streamed message: each block is started, filled by deltas, and stopped."""
    usage = usage or {"input_tokens": 100, "output_tokens": 20}
    events: list[dict] = [{
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {**usage, "output_tokens": 0}},
    }]  # fmt: skip
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            events.append({"type": "content_block_start", "index": i,
                           "content_block": {"type": "text", "text": ""}})  # fmt: skip
            for piece in (block["text"][:5], block["text"][5:]):
                events.append({"type": "content_block_delta", "index": i,
                               "delta": {"type": "text_delta", "text": piece}})  # fmt: skip
        elif block["type"] == "thinking":
            events.append({"type": "content_block_start", "index": i,
                           "content_block": {"type": "thinking", "thinking": "",
                                             "signature": ""}})  # fmt: skip
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "signature_delta",
                                     "signature": block["signature"]}})  # fmt: skip
        elif block["type"] == "tool_use":
            events.append({"type": "content_block_start", "index": i,
                           "content_block": {"type": "tool_use", "id": block["id"],
                                             "name": block["name"], "input": {}}})  # fmt: skip
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "input_json_delta",
                                     "partial_json": json.dumps(block["input"])}})  # fmt: skip
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                   "usage": {"output_tokens": usage["output_tokens"]}})  # fmt: skip
    events.append({"type": "message_stop"})
    return sse(*events)


class Recorder:
    def __init__(self, *responses: httpx2.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def provider(recorder: Recorder, **kwargs) -> AnthropicProvider:
    client = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(recorder))
    return AnthropicProvider(api_key="test-key", http_client=client, **kwargs)


def streamed(body: bytes) -> httpx2.Response:
    return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})


def run(p: AnthropicProvider, request: Request) -> list:
    return list(p.stream(request, CancelToken()))


PATH_SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}}
TOOLS = (ToolSpec("read_file", "Read a file", PATH_SCHEMA),)


def test_streams_text_tool_calls_and_cached_usage():
    rec = Recorder(streamed(message_events(
        [{"type": "thinking", "signature": "sig-abc"},
         {"type": "text", "text": "Let me look at that file."},
         {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}],
        "tool_use",
        {"input_tokens": 40, "output_tokens": 30, "cache_read_input_tokens": 900,
         "cache_creation_input_tokens": 60},
    )))  # fmt: skip
    events = run(provider(rec), Request(MODEL, (Message("user", "fix it"),), TOOLS))

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == (
        "Let me look at that file."
    )
    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert calls == [ToolCall("toolu_1", "read_file", {"path": "a.py"})]
    done = events[-1]
    assert isinstance(done, Done) and done.stop_reason == "tool_calls"
    assert done.usage == Usage(1000, 30, cache_read_tokens=900, cache_write_tokens=60)
    # The thinking block (with its signature) is kept for verbatim replay.
    kinds = [b["type"] for b in done.provider_state["content"]]
    assert kinds == ["thinking", "text", "tool_use"]
    assert done.provider_state["content"][0]["signature"] == "sig-abc"


def test_request_shape():
    rec = Recorder(streamed(message_events([{"type": "text", "text": "ok"}], "end_turn")))
    messages = (
        Message("system", "SYSTEM"),
        Message("user", "task"),
        Message("assistant", "", (ToolCall("t1", "read_file", {"path": "a"}),
                                  ToolCall("t2", "read_file", {"path": "b"}))),
        Message("tool", "A", tool_call_id="t1", tool_name="read_file"),
        Message("tool", "B", tool_call_id="t2", tool_name="read_file"),
        Message("user", "note after results"),
    )  # fmt: skip
    run(provider(rec, effort="medium"), Request(MODEL, messages, TOOLS, max_output_tokens=5000))
    body = rec.body()

    assert body["model"] == MODEL and body["max_tokens"] == 5000
    assert body["system"] == [{"type": "text", "text": "SYSTEM"}]
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert body["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert "temperature" not in body
    assert body["tools"][0]["input_schema"]["properties"]["path"] == {"type": "string"}
    # Both results, then the note, in ONE user turn right after the tool_use turn.
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    last = body["messages"][-1]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"]
    assert [b.get("tool_use_id") for b in last[:2]] == ["t1", "t2"]
    betas = rec.requests[-1].headers["anthropic-beta"]
    assert FALLBACK_BETA in betas and THINKING_BINDING_BETA in betas
    assert rec.requests[-1].headers["x-api-key"] == "test-key"


def test_options_off_send_plain_request():
    rec = Recorder(streamed(message_events([{"type": "text", "text": "ok"}], "end_turn")))
    p = provider(rec, refusal_fallback=False, drop_mismatched_thinking=False)
    run(p, Request(MODEL, (Message("user", "hi"),)))
    body = rec.body()
    assert not {"fallbacks", "thinking", "output_config", "tools"} & set(body)
    assert "anthropic-beta" not in rec.requests[-1].headers


def test_assistant_state_is_replayed_verbatim():
    state = {"content": [{"type": "thinking", "thinking": "", "signature": "sig"},
                         {"type": "tool_use", "id": "t1", "name": "read_file",
                          "input": {"path": "a"}}]}  # fmt: skip
    rec = Recorder(streamed(message_events([{"type": "text", "text": "ok"}], "end_turn")))
    messages = (
        Message("user", "task"),
        Message("assistant", "", (ToolCall("t1", "read_file", {"path": "a"}),),
                provider_state=state),
        Message("tool", "A", tool_call_id="t1", tool_name="read_file"),
    )  # fmt: skip
    run(provider(rec), Request(MODEL, messages, TOOLS))
    assert rec.body()["messages"][1]["content"] == state["content"]


def test_refusal_never_yields_tool_calls():
    rec = Recorder(streamed(message_events(
        [{"type": "text", "text": "partial"},
         {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a"}}],
        "refusal",
    )))  # fmt: skip
    events = run(provider(rec), Request(MODEL, (Message("user", "x"),), TOOLS))
    assert not [e for e in events if isinstance(e, ToolCallEvent)]
    done = events[-1]
    assert done.stop_reason == "refusal"
    assert all(b["type"] != "tool_use" for b in done.provider_state["content"])


def test_replay_drops_model_internal_blocks_before_a_fallback_boundary():
    content = [
        {"type": "thinking", "signature": "a"},
        {"type": "text", "text": "partial answer"},
        {"type": "tool_use", "id": "x"},
        {"type": "fallback", "from": {"model": "m1"}, "to": {"model": "m2"}},
        {"type": "thinking", "signature": "b"},
        {"type": "text", "text": "continued"},
    ]
    assert [b["type"] for b in _replayable(content)] == ["text", "thinking", "text"]
    assert _replayable(content[:3]) == content[:3]  # no fallback: unchanged


@pytest.mark.parametrize(
    ("status", "retryable"), [(529, True), (503, True), (429, True), (400, False), (401, False)]
)
def test_http_errors_map_to_provider_errors(status, retryable):
    error = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
    rec = Recorder(httpx2.Response(status, json=error))
    with pytest.raises(ProviderError) as info:
        run(provider(rec), Request(MODEL, (Message("user", "x"),)))
    assert info.value.retryable is retryable and info.value.status == status
    assert len(rec.requests) == 1  # the SDK's own retries are off; the gateway owns retries


def test_missing_key_is_not_retryable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError) as info:
        run(AnthropicProvider(), Request(MODEL, (Message("user", "x"),)))
    assert not info.value.retryable and "ANTHROPIC_API_KEY" in str(info.value)


def test_cancel_stops_the_stream():
    rec = Recorder(streamed(message_events([{"type": "text", "text": "hello world"}], "end_turn")))
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(Exception, match="cancelled"):
        list(provider(rec).stream(Request(MODEL, (Message("user", "x"),)), cancel))


def test_list_models_reports_context_window():
    page = {"data": [{"id": MODEL, "type": "model", "display_name": "Test",
                      "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1_000_000,
                      "max_tokens": 128_000}],
            "has_more": False, "first_id": MODEL, "last_id": MODEL}  # fmt: skip
    models = provider(Recorder(httpx2.Response(200, json=page))).list_models()
    assert [(m.name, m.input_token_limit, m.output_token_limit) for m in models] == [
        (MODEL, 1_000_000, 128_000)
    ]


def test_cache_aware_pricing():
    usage = Usage(1_000_000, 100_000, cache_read_tokens=800_000, cache_write_tokens=100_000)
    priced = Price(4.0, 20.0, cache_read_per_mtok=0.2, cache_write_per_mtok=5.0)
    # 100k uncached * 4 + 800k * 0.2 + 100k * 5 + 100k out * 20, per million
    assert priced.cost(usage) == pytest.approx(0.4 + 0.16 + 0.5 + 2.0)
    # Unconfigured cache rates can only overstate: reads at full input, writes at 1.25x.
    assert Price(4.0, 20.0).cost(usage) == pytest.approx(0.4 + 3.2 + 0.5 + 2.0)


def test_factory_builds_anthropic_provider_from_config():
    cfg = LLMConfig.model_validate({
        "providers": {"claude": {"kind": "anthropic", "effort": "low"}},
        "routes": {"strong": [f"claude:{MODEL}"]},
        "pricing": {f"claude:{MODEL}": {"input_per_mtok": 1, "output_per_mtok": 2,
                                         "cache_read_per_mtok": 0.1}},
    })  # fmt: skip
    gw = build_gateway(cfg)
    p = gw.providers["claude"]
    assert isinstance(p, AnthropicProvider)
    assert p.api_key_env == "ANTHROPIC_API_KEY" and p.effort == "low" and p.refusal_fallback
    assert gw.prices[f"claude:{MODEL}"].cache_read_per_mtok == 0.1
