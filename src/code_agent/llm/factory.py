"""Build a Gateway from config."""

from __future__ import annotations

from collections.abc import Callable

from code_agent.config import LLMConfig
from code_agent.llm.gateway import Gateway, Price, RetryPolicy, Route
from code_agent.llm.providers import FakeProvider, GeminiProvider, Provider
from code_agent.llm.types import Request


def build_providers(cfg: LLMConfig) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    for name, p in cfg.providers.items():
        if p.kind == "gemini":
            providers[name] = GeminiProvider(
                api_key_env=p.api_key_env, base_url=p.base_url, timeout_s=p.timeout_s, name=name
            )
        elif p.kind == "fake":
            if p.script is None:
                raise ValueError(f"provider {name!r}: kind='fake' needs a script file")
            providers[name] = FakeProvider.from_file(p.script, name=name)
    return providers


def build_gateway(
    cfg: LLMConfig,
    *,
    providers: dict[str, Provider] | None = None,
    outbound_filter: Callable[[Request], Request] | None = None,
) -> Gateway:
    return Gateway(
        providers=providers if providers is not None else build_providers(cfg),
        routes={role: [Route.parse(s) for s in specs] for role, specs in cfg.routes.items()},
        prices={
            spec: Price(p.input_per_mtok, p.output_per_mtok) for spec, p in cfg.pricing.items()
        },
        retry=RetryPolicy(cfg.max_attempts, cfg.base_delay_s, cfg.max_delay_s),
        outbound_filter=outbound_filter,
    )
