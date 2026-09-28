from __future__ import annotations

import random

import pytest

from code_agent.config import LLMConfig
from code_agent.llm.factory import build_gateway
from code_agent.llm.gateway import (
    Gateway,
    GenerationUnavailableError,
    ModelUnavailableError,
    Price,
    RetryPolicy,
    Route,
)
from code_agent.llm.providers import FakeProvider, FakeTurn
from code_agent.llm.types import (
    CancelledError,
    CancelToken,
    Done,
    Message,
    ProviderError,
    Request,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
)

MSGS = [Message("system", "be brief"), Message("user", "hi")]
NO_WAIT = RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)


def gateway(*providers: FakeProvider, routes: dict[str, list[str]], **kw) -> Gateway:
    return Gateway(
        providers={p.name: p for p in providers},
        routes={role: [Route.parse(s) for s in specs] for role, specs in routes.items()},
        retry=kw.pop("retry", NO_WAIT),
        **kw,
    )


def text_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def test_route_parsing():
    assert Route.parse("gemini:some-model-1") == Route("gemini", "some-model-1")
    with pytest.raises(ValueError):
        Route.parse("no-provider")


def test_streams_text_tool_calls_and_done():
    call = ToolCall("c1", "read_file", {"path": "a.py"})
    fake = FakeProvider([FakeTurn("hello world", (call,), Usage(10, 5))])
    gw = gateway(fake, routes={"strong": ["fake:fake-strong"]})
    events = list(gw.stream("strong", MSGS))
    assert text_of(events) == "hello world"
    assert [e.call for e in events if isinstance(e, ToolCallEvent)] == [call]
    assert isinstance(events[-1], Done) and events[-1].stop_reason == "tool_calls"
    assert fake.requests[0].model == "fake-strong"


def test_role_routing_uses_cheap_and_strong_models():
    fake = FakeProvider([FakeTurn("a"), FakeTurn("b")])
    gw = gateway(fake, routes={"cheap": ["fake:fake-cheap"], "strong": ["fake:fake-strong"]})
    list(gw.stream("cheap", MSGS))
    list(gw.stream("strong", MSGS))
    assert [r.model for r in fake.requests] == ["fake-cheap", "fake-strong"]


def test_retryable_errors_are_retried_then_succeed():
    flaky = ProviderError("429 slow down", retryable=True, status=429)
    fake = FakeProvider([FakeTurn(error=flaky), FakeTurn(error=flaky), FakeTurn("ok")])
    gw = gateway(fake, routes={"strong": ["fake:fake-strong"]})
    assert text_of(gw.stream("strong", MSGS)) == "ok"
    assert gw.last_call is not None and gw.last_call.attempts == 3


def test_falls_back_to_next_route_after_retries_exhausted():
    down = ProviderError("503", retryable=True, status=503)
    primary = FakeProvider([FakeTurn(error=down)] * 3, name="primary")
    backup = FakeProvider([FakeTurn("from backup")], name="backup")
    gw = gateway(primary, backup, routes={"strong": ["primary:fake-strong", "backup:fake-cheap"]})
    assert text_of(gw.stream("strong", MSGS)) == "from backup"
    assert gw.last_call is not None and str(gw.last_call.route) == "backup:fake-cheap"
    assert gw.last_call.attempts == 4


def test_non_retryable_error_skips_straight_to_fallback():
    bad = ProviderError("400 bad request", retryable=False, status=400)
    primary = FakeProvider([FakeTurn(error=bad)], name="primary")
    backup = FakeProvider([FakeTurn("ok")], name="backup")
    gw = gateway(primary, backup, routes={"strong": ["primary:m", "backup:m"]})
    assert text_of(gw.stream("strong", MSGS)) == "ok"
    assert len(primary.requests) == 1


def test_everything_down_raises_generation_unavailable():
    down = ProviderError("503", retryable=True)
    fake = FakeProvider([FakeTurn(error=down)] * 3)
    gw = gateway(fake, routes={"strong": ["fake:fake-strong"]})
    with pytest.raises(GenerationUnavailableError) as exc:
        list(gw.stream("strong", MSGS))
    assert "generation unavailable" in str(exc.value) and len(exc.value.failures) == 3


def test_unconfigured_role_is_a_clear_error():
    gw = gateway(FakeProvider(), routes={})
    with pytest.raises(GenerationUnavailableError, match="no routes configured"):
        list(gw.stream("strong", MSGS))


def test_error_after_partial_output_is_not_retried_silently():
    class DiesMidStream(FakeProvider):
        def stream(self, request, cancel):
            self.requests.append(request)
            yield TextDelta("partial ")
            raise ProviderError("connection reset", retryable=True)

    fake = DiesMidStream(name="fake")
    gw = gateway(fake, routes={"strong": ["fake:m"]})
    with pytest.raises(ProviderError):
        list(gw.stream("strong", MSGS))
    assert len(fake.requests) == 1


def test_backoff_uses_full_jitter_within_cap():
    policy = RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=5.0)
    rng = random.Random(0)
    delays = [policy.delay(a, rng) for a in range(5) for _ in range(200)]
    assert all(0 <= d <= 5.0 for d in delays)
    assert max(policy.delay(0, rng) for _ in range(200)) <= 1.0


def test_cancel_during_backoff():
    down = ProviderError("503", retryable=True)
    fake = FakeProvider([FakeTurn(error=down)] * 3)
    cancel = CancelToken()
    gw = gateway(fake, routes={"strong": ["fake:m"]},
                 retry=RetryPolicy(max_attempts=3, base_delay_s=30, max_delay_s=30),
                 rng=random.Random(1))  # fmt: skip
    cancel.cancel()
    with pytest.raises(CancelledError):
        list(gw.stream("strong", MSGS, cancel=cancel))


def test_cost_accounting_and_unknown_price():
    fake = FakeProvider([FakeTurn("a", usage=Usage(1_000_000, 100_000)), FakeTurn("b")])
    gw = gateway(fake, routes={"strong": ["fake:priced"], "cheap": ["fake:unpriced"]},
                 prices={"fake:priced": Price(1.0, 4.0)})  # fmt: skip
    list(gw.stream("strong", MSGS))
    assert gw.last_call is not None and gw.last_call.cost_usd == pytest.approx(1.4)
    assert gw.total_cost_usd == pytest.approx(1.4)
    list(gw.stream("cheap", MSGS))
    assert gw.last_call.cost_usd is None
    assert gw.total_cost_usd is None  # an unpriced call makes the total unknown, not 1.4
    assert gw.total_usage == Usage(1_001_000, 100_200)


def test_outbound_filter_sees_and_rewrites_every_request():
    fake = FakeProvider([FakeTurn("ok")])

    def redact(req: Request) -> Request:
        msgs = tuple(
            Message(m.role, m.content.replace("hunter2", "[REDACTED]")) for m in req.messages
        )
        return Request(req.model, msgs, req.tools, req.max_output_tokens, req.temperature)

    gw = gateway(fake, routes={"strong": ["fake:m"]}, outbound_filter=redact)
    list(gw.stream("strong", [Message("user", "password is hunter2")]))
    assert fake.requests[0].messages[0].content == "password is [REDACTED]"


def test_verify_models_accepts_known_and_reports_unknown_with_alternatives():
    fake = FakeProvider()
    gw = gateway(fake, routes={"strong": ["fake:fake-strong"], "cheap": ["fake:fake-cheap"]})
    gw.verify_models()
    assert gw.context_window("strong") == 128_000
    bad = gateway(fake, routes={"strong": ["fake:retired-model"], "x": ["nope:m"]})
    with pytest.raises(ModelUnavailableError) as exc:
        bad.verify_models()
    assert "retired-model" in str(exc.value) and "fake-strong" in str(exc.value)
    assert "unknown provider 'nope'" in str(exc.value)


def test_context_window_is_smallest_across_fallbacks():
    gw = gateway(FakeProvider(), routes={"strong": ["fake:fake-strong", "fake:fake-cheap"]})
    gw.verify_models()
    assert gw.context_window("strong") == 32_000


def test_build_gateway_from_config():
    cfg = LLMConfig.model_validate(
        {
            "routes": {"strong": ["fake:fake-strong"]},
            "pricing": {"fake:fake-strong": {"input_per_mtok": 1, "output_per_mtok": 2}},
            "max_attempts": 2,
        }
    )
    fake = FakeProvider([FakeTurn("x")])
    gw = build_gateway(cfg, providers={"fake": fake})
    assert gw.routes["strong"] == [Route("fake", "fake-strong")]
    assert gw.retry.max_attempts == 2
    assert text_of(gw.stream("strong", MSGS)) == "x"
