"""The one place the agent talks to LLMs.

Responsibilities:
* **Routing by role.** The loop asks for a *role* ("cheap" for planning and query rewriting,
  "strong" for edits), never a model name. Config maps each role to an ordered list of
  `provider:model` routes; later routes are fallbacks.
* **Model verification.** Before first use, every configured model is checked against the
  provider's live model list, so a retired or misspelled model fails fast with the list of
  models that do exist, instead of failing mid-task.
* **Retries.** Retryable errors (timeouts, 429, 5xx) back off exponentially with full jitter;
  a route that keeps failing (or fails non-retryably) falls through to the next route. A call
  that already streamed output is *not* retried transparently, because the caller has seen
  partial text; the error surfaces instead.
* **Outbound filter.** Every request passes through `outbound_filter` just before it is sent.
  M3 plugs secret redaction in here, so no code path can bypass it.
* **Accounting.** Token usage and cost per call (`last_call`) and in total. Cost is `None` when
  a model has no configured price: reported as unknown, never guessed.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from code_agent.llm.providers import ModelInfo, Provider
from code_agent.llm.types import (
    CancelledError,
    CancelToken,
    Done,
    Message,
    ProviderError,
    Request,
    StreamEvent,
    ToolSpec,
    Usage,
)


@dataclass(frozen=True)
class Route:
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> Route:
        provider, sep, model = spec.partition(":")
        if not sep or not provider or not model:
            raise ValueError(f"route {spec!r} must look like 'provider:model'")
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class Price:
    input_per_mtok: float  # USD per million input tokens
    output_per_mtok: float
    # Prompt-cache rates. Unset: reads are charged as normal input and writes at 1.25x input,
    # so an unconfigured cache price can only overstate the cost, never understate it.
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def cost(self, usage: Usage) -> float:
        read = self.input_per_mtok if self.cache_read_per_mtok is None else self.cache_read_per_mtok
        write = (
            self.input_per_mtok * 1.25
            if self.cache_write_per_mtok is None
            else self.cache_write_per_mtok
        )
        uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
        return (
            uncached * self.input_per_mtok
            + usage.cache_read_tokens * read
            + usage.cache_write_tokens * write
            + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4  # per route
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0

    def delay(self, attempt: int, rng: random.Random) -> float:
        """Full jitter: uniform(0, min(cap, base * 2**attempt)). Spreads retries from many
        clients so they don't hit a recovering provider in lockstep."""
        return rng.uniform(0, min(self.max_delay_s, self.base_delay_s * 2**attempt))


@dataclass(frozen=True)
class CallRecord:
    role: str
    route: Route
    usage: Usage
    cost_usd: float | None
    attempts: int  # total provider calls made, across retries and fallbacks
    seconds: float


class ModelUnavailableError(RuntimeError):
    pass


class GenerationUnavailableError(RuntimeError):
    """Every route for a role failed. The agent reports this and keeps offline features working."""

    def __init__(self, role: str, failures: list[str]) -> None:
        super().__init__(f"generation unavailable for role {role!r}: " + "; ".join(failures[-4:]))
        self.failures = failures


@dataclass
class Gateway:
    providers: Mapping[str, Provider]
    routes: Mapping[str, Sequence[Route]]
    prices: Mapping[str, Price] = field(default_factory=dict)  # key: "provider:model"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    outbound_filter: Callable[[Request], Request] | None = None
    rng: random.Random = field(default_factory=random.Random)
    last_call: CallRecord | None = None
    total_usage: Usage = field(default_factory=Usage)
    total_cost_usd: float | None = 0.0
    _models: dict[str, ModelInfo] = field(default_factory=dict)

    # -- model verification -------------------------------------------------------------------

    def verify_models(self) -> dict[str, ModelInfo]:
        """Check every configured route against the provider's model list. Returns route->info."""
        available: dict[str, dict[str, ModelInfo]] = {}
        problems: list[str] = []
        for role, routes in self.routes.items():
            for route in routes:
                provider = self.providers.get(route.provider)
                if provider is None:
                    problems.append(f"{role}: unknown provider {route.provider!r}")
                    continue
                if route.provider not in available:
                    available[route.provider] = {m.name: m for m in provider.list_models()}
                info = available[route.provider].get(route.model)
                if info is None:
                    names = ", ".join(sorted(available[route.provider])[:15])
                    problems.append(
                        f"{role}: {route} is not offered by {route.provider} (available: {names})"
                    )
                else:
                    self._models[str(route)] = info
        if problems:
            raise ModelUnavailableError("; ".join(problems))
        return dict(self._models)

    def context_window(self, role: str) -> int | None:
        """Smallest context window among a role's routes (a fallback must fit the same prompt)."""
        limits = [
            info.input_token_limit
            for r in self.routes[role]
            if (info := self._models.get(str(r))) and info.input_token_limit
        ]
        return min(limits) if limits else None

    # -- calls --------------------------------------------------------------------------------

    def stream(
        self,
        role: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_output_tokens: int = 4096,
        cancel: CancelToken | None = None,
    ) -> Iterator[StreamEvent]:
        cancel = cancel or CancelToken()
        if role not in self.routes or not self.routes[role]:
            raise GenerationUnavailableError(role, [f"no routes configured for role {role!r}"])
        failures: list[str] = []
        attempts = 0
        started = time.perf_counter()
        for route in self.routes[role]:
            provider = self.providers.get(route.provider)
            if provider is None:
                failures.append(f"{route}: unknown provider")
                continue
            for attempt in range(self.retry.max_attempts):
                cancel.raise_if_cancelled()
                attempts += 1
                request = Request(route.model, tuple(messages), tuple(tools), max_output_tokens)
                if self.outbound_filter is not None:
                    request = self.outbound_filter(request)
                emitted = False
                try:
                    for event in provider.stream(request, cancel):
                        emitted = True
                        if isinstance(event, Done):
                            self._account(role, route, event.usage, attempts, started)
                        yield event
                    return
                except ProviderError as exc:
                    failures.append(f"{route} attempt {attempt + 1}: {exc}")
                    if emitted:
                        raise  # partial output already delivered; the caller decides
                    if not exc.retryable or attempt == self.retry.max_attempts - 1:
                        break  # fall back to the next route
                    if cancel.wait(self.retry.delay(attempt, self.rng)):
                        raise CancelledError("cancelled during retry backoff") from exc
        raise GenerationUnavailableError(role, failures)

    def _account(
        self, role: str, route: Route, usage: Usage, attempts: int, started: float
    ) -> None:
        price = self.prices.get(str(route))
        cost = price.cost(usage) if price else None
        self.last_call = CallRecord(
            role, route, usage, cost, attempts, time.perf_counter() - started
        )
        self.total_usage = self.total_usage + usage
        if cost is None or self.total_cost_usd is None:
            self.total_cost_usd = None  # once any call is unpriced, the total is unknown
        else:
            self.total_cost_usd += cost
