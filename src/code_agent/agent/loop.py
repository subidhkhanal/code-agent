"""The agent loop: one task from request to a validated (not yet applied) change set.

    request -> query rewrite (cheap model) -> context assembly -> [strong model turn
      -> tool calls? execute, feed back as untrusted data, loop
      -> edit blocks? Fast Apply plan; rejected -> feedback + retry (bounded); ok -> done
      -> neither? the model answered in prose -> done]

The loop never writes to the workspace. It returns an `ApplyPlan`; applying it is the caller's
decision (after human review in chat, or after shadow validation in M3).

Budgets are checked in code before every model call and every tool call. Exhausting one ends
the task with a clear reason and whatever partial result exists; nothing the model says can
change a budget.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from code_agent.agent.prompts import (
    REWRITE_PROMPT,
    SYSTEM_PROMPT,
    rejection_message,
    task_message,
)
from code_agent.agent.tasks import TaskStatus, TaskUsage
from code_agent.agent.tools import ToolRegistry
from code_agent.config import BudgetConfig
from code_agent.context.assembly import ContextAssembler, ContextBundle, estimate_tokens
from code_agent.edits.apply import ApplyPlan, Planner
from code_agent.edits.parser import EditBlock, ParseError, StreamingEditParser, Text
from code_agent.llm.gateway import Gateway, GenerationUnavailableError
from code_agent.llm.types import (
    CancelledError,
    CancelToken,
    Done,
    Message,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)

ELIDED = "[older tool output elided to stay within the context budget]"


# -- events for the UI ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Status:
    message: str


@dataclass(frozen=True)
class RetrievalDone:
    bundle: ContextBundle


@dataclass(frozen=True)
class AssistantText:
    text: str


@dataclass(frozen=True)
class EditProposed:
    block: EditBlock


@dataclass(frozen=True)
class ToolStarted:
    call: ToolCall


@dataclass(frozen=True)
class ToolFinished:
    call: ToolCall
    is_error: bool
    summary: str


@dataclass(frozen=True)
class EditsRejected:
    feedback: str
    attempt: int


AgentEvent = (
    Status
    | RetrievalDone
    | AssistantText
    | EditProposed
    | ToolStarted
    | ToolFinished
    | EditsRejected
)


# -- budgets -------------------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class Meter:
    budgets: BudgetConfig
    usage: TaskUsage = field(default_factory=TaskUsage)
    started: float = field(default_factory=time.monotonic)

    def check(self) -> None:
        b, u = self.budgets, self.usage
        if u.tokens >= b.max_tokens:
            raise BudgetExceededError(f"token budget exhausted ({u.tokens:,}/{b.max_tokens:,})")
        if b.max_usd is not None and u.cost_usd is not None and u.cost_usd >= b.max_usd:
            raise BudgetExceededError(f"cost budget exhausted (${u.cost_usd:.4f}/${b.max_usd})")
        if u.tool_calls >= b.max_tool_calls:
            raise BudgetExceededError(f"tool-call budget exhausted ({u.tool_calls})")
        elapsed = time.monotonic() - self.started
        if elapsed >= b.max_seconds:
            raise BudgetExceededError(f"time budget exhausted ({elapsed:.0f}s/{b.max_seconds}s)")

    def add_call(self, gateway: Gateway) -> None:
        call = gateway.last_call
        if call is None:
            return
        self.usage.llm_calls += 1
        self.usage.input_tokens += call.usage.input_tokens
        self.usage.output_tokens += call.usage.output_tokens
        if call.cost_usd is None or self.usage.cost_usd is None:
            self.usage.cost_usd = None  # unknown price: USD budget can't be enforced
        else:
            self.usage.cost_usd += call.cost_usd


# -- result --------------------------------------------------------------------------------------


@dataclass
class TaskResult:
    status: TaskStatus
    message: str  # why the task ended, for the user
    plan: ApplyPlan | None = None  # validated edits (not applied); None if no edits
    answer: str = ""  # the model's final prose
    usage: TaskUsage = field(default_factory=TaskUsage)
    rejections: list[str] = field(default_factory=list)  # ApplyErrorCode / parse error per block
    bundle: ContextBundle | None = None


@dataclass
class _Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    blocks: list[EditBlock] = field(default_factory=list)
    parse_errors: list[ParseError] = field(default_factory=list)
    provider_state: dict | None = None


_QUERY_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*")


class AgentLoop:
    def __init__(
        self,
        gateway: Gateway,
        tools: ToolRegistry,
        assembler: ContextAssembler,
        planner: Planner,
        budgets: BudgetConfig,
        *,
        context_budget_tokens: int,
        history_budget_tokens: int,
        max_output_tokens: int = 8192,
        on_event: Callable[[AgentEvent], None] = lambda _: None,
    ) -> None:
        self.gateway = gateway
        self.tools = tools
        self.assembler = assembler
        self.planner = planner
        self.budgets = budgets
        self.context_budget_tokens = context_budget_tokens
        self.history_budget_tokens = history_budget_tokens
        self.max_output_tokens = max_output_tokens
        self.emit = on_event

    def run(self, task: str, cancel: CancelToken | None = None) -> TaskResult:
        cancel = cancel or CancelToken()
        meter = Meter(self.budgets)
        result = TaskResult(TaskStatus.RUNNING, "", usage=meter.usage)
        try:
            self._run(task, cancel, meter, result)
        except CancelledError:
            result.status, result.message = TaskStatus.CANCELLED, "cancelled"
        except BudgetExceededError as exc:
            result.status, result.message = TaskStatus.FAILED, str(exc)
        except (GenerationUnavailableError, ProviderError) as exc:
            result.status, result.message = TaskStatus.FAILED, str(exc)
        return result

    # -- phases -------------------------------------------------------------------------------

    def _run(self, task: str, cancel: CancelToken, meter: Meter, result: TaskResult) -> None:
        queries = self._rewrite(task, cancel, meter)
        bundle = self.assembler.assemble(queries, self.context_budget_tokens)
        result.bundle = bundle
        self.tools.ctx.base_hashes.update(bundle.base_hashes)
        self.emit(RetrievalDone(bundle))

        messages = [
            Message("system", SYSTEM_PROMPT),
            Message("user", task_message(task, bundle.render())),
        ]
        while True:
            cancel.raise_if_cancelled()
            meter.check()
            self._fit_history(messages)
            turn = self._turn(messages, cancel, meter)
            messages.append(
                Message(
                    "assistant",
                    turn.text,
                    tuple(turn.tool_calls),
                    provider_state=turn.provider_state,
                )
            )
            result.answer = turn.text

            if turn.tool_calls:
                for call in turn.tool_calls:
                    cancel.raise_if_cancelled()
                    meter.check()
                    meter.usage.tool_calls += 1
                    self.emit(ToolStarted(call))
                    outcome = self.tools.execute(call)
                    self.emit(ToolFinished(call, outcome.is_error, outcome.content[:200]))
                    messages.append(
                        Message("tool", outcome.render(), tool_call_id=call.id, tool_name=call.name)
                    )
                if turn.blocks:
                    messages.append(Message("user", "Edit blocks sent together with tool calls "
                                            "were ignored. Send them again once you have what "
                                            "you need."))  # fmt: skip
                continue

            if not turn.blocks and not turn.parse_errors:
                result.status, result.message = TaskStatus.SUCCEEDED, "answered without edits"
                return

            plan = self.planner.plan(turn.blocks, self.tools.ctx.base_hashes)
            feedback = "\n".join(e.message for e in turn.parse_errors)
            if plan.ok and not turn.parse_errors:
                result.plan = plan
                result.status, result.message = TaskStatus.SUCCEEDED, "edits ready for review"
                return

            meter.usage.edit_attempts += 1
            result.rejections += [str(r.error) for r in plan.errors] + ["PARSE_ERROR"] * len(
                turn.parse_errors
            )
            feedback = "\n\n".join(x for x in (feedback, plan.feedback()) if x)
            self.emit(EditsRejected(feedback, meter.usage.edit_attempts))
            left = self.budgets.max_edit_attempts - meter.usage.edit_attempts
            if left <= 0:
                result.status = TaskStatus.FAILED
                result.message = f"edits still rejected after {meter.usage.edit_attempts} attempts"
                return
            messages.append(Message("user", rejection_message(feedback, left)))

    def _rewrite(self, task: str, cancel: CancelToken, meter: Meter) -> list[str]:
        """Cheap model turns the request into search queries. Falls back to the raw request."""
        try:
            meter.check()
            text = "".join(
                e.text
                for e in self.gateway.stream(
                    "cheap", [Message("user", REWRITE_PROMPT.format(task=task))],
                    max_output_tokens=200, cancel=cancel,
                )
                if isinstance(e, TextDelta)
            )  # fmt: skip
            meter.add_call(self.gateway)
        except (GenerationUnavailableError, ProviderError) as exc:
            self.emit(Status(f"query rewrite unavailable ({exc}); searching with the request"))
            return [task]
        rewrites = [_QUERY_LINE.sub("", line).strip() for line in text.splitlines()]
        return [task, *[q for q in rewrites if q][:3]]

    def _turn(self, messages: list[Message], cancel: CancelToken, meter: Meter) -> _Turn:
        turn = _Turn()
        parser = StreamingEditParser()
        parts: list[str] = []

        def handle(events) -> None:
            for event in events:
                if isinstance(event, EditBlock):
                    turn.blocks.append(event)
                    self.emit(EditProposed(event))
                elif isinstance(event, ParseError):
                    turn.parse_errors.append(event)
                elif isinstance(event, Text):
                    self.emit(AssistantText(event.text))

        for event in self.gateway.stream(
            "strong", messages, self.tools.specs(),
            max_output_tokens=self.max_output_tokens, cancel=cancel,
        ):  # fmt: skip
            if isinstance(event, TextDelta):
                parts.append(event.text)
                handle(parser.feed(event.text))
            elif isinstance(event, ToolCallEvent):
                turn.tool_calls.append(event.call)
            elif isinstance(event, Done):
                turn.provider_state = event.provider_state
        handle(parser.finish())
        meter.add_call(self.gateway)
        turn.text = "".join(parts)
        return turn

    def _fit_history(self, messages: list[Message]) -> None:
        """Keep the whole prompt under the history budget by eliding the oldest tool outputs.
        The system prompt, the task (with retrieved code) and the latest turn are kept."""

        def total() -> int:
            return sum(estimate_tokens(m.content) for m in messages)

        for i, message in enumerate(messages[:-2]):
            if total() <= self.history_budget_tokens:
                return
            if message.role == "tool" and message.content != ELIDED:
                messages[i] = Message("tool", ELIDED, tool_call_id=message.tool_call_id,
                                      tool_name=message.tool_name)  # fmt: skip
