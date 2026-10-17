"""One playground run: copy a sample repo, run the normal agent on it, report what happened.

The engine is the same one `agent chat` uses. What differs is policy, all enforced in code:
* Each run gets a fresh copy of the sample in a temp dir, so visitors never see each other's
  edits, and the copy is deleted afterwards.
* Commands: the model may run test/lint commands (each approved once, automatically, and
  audited). Anything else, including read-only shell commands and everything privileged, is
  denied. Visitors can't widen this; there is no approval prompt to talk them into.
* Budgets: per-run cost, time and tool-call caps from the playground settings override the
  config file's.
* Edits are validated in the shadow worktree (lint, type check, tests) and shown as a diff.
  They are never applied: there is nothing to apply them to once the run's copy is deleted.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from code_agent.agent.loop import (
    AgentEvent,
    AssistantText,
    EditProposed,
    EditsRejected,
    RetrievalDone,
    Status,
    ToolFinished,
    ToolStarted,
    ValidationDone,
    ValidationStarted,
)
from code_agent.agent.session import AgentSession
from code_agent.config import AgentConfig
from code_agent.edits.diff import plan_diff
from code_agent.headless import validation_dict
from code_agent.index.embeddings import Embedder
from code_agent.llm.factory import build_gateway
from code_agent.llm.providers import ModelInfo, Provider
from code_agent.llm.types import CancelToken, Request
from code_agent.security.approvals import ApprovalPrompt, Scope
from code_agent.security.commands import Category
from code_agent.workspace import Workspace, run_git

SAMPLES_DIR = Path(__file__).parent / "samples"
# A path line followed by one SEARCH/REPLACE block; the diff already shows these.
_EDIT_BLOCK = re.compile(r"^[^\n]*\n<<<<<<< SEARCH\n.*?^>>>>>>> REPLACE[^\n]*\n?", re.M | re.S)

Emit = Callable[[dict], None]


@dataclass(frozen=True)
class Sample:
    id: str
    title: str
    blurb: str
    tasks: list[str]

    @property
    def path(self) -> Path:
        return SAMPLES_DIR / self.id

    def files(self) -> dict[str, str]:
        """Source files for the code browser (small repos only, by construction)."""
        out: dict[str, str] = {}
        for p in sorted(self.path.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                out[p.relative_to(self.path).as_posix()] = p.read_text(encoding="utf-8")
        return out


def load_samples() -> dict[str, Sample]:
    catalog = json.loads((SAMPLES_DIR / "catalog.json").read_text(encoding="utf-8"))
    return {s["id"]: Sample(**s) for s in catalog}


@dataclass
class CachedModels:
    """Wraps a provider so the live model list is fetched once per process, not once per run."""

    inner: Provider
    name: str = ""
    _models: list[ModelInfo] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.name = self.inner.name

    def stream(self, request: Request, cancel: CancelToken):
        return self.inner.stream(request, cancel)

    def list_models(self) -> list[ModelInfo]:
        with self._lock:
            if self._models is None:
                self._models = self.inner.list_models()
            return list(self._models)


class LockedEmbedder:
    """Serializes embedding calls: one model instance is shared by concurrent runs."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed_documents(self, texts):
        with self._lock:
            return self.inner.embed_documents(texts)

    def embed_query(self, text):
        with self._lock:
            return self.inner.embed_query(text)


def playground_approver(emit: Emit) -> Callable[[ApprovalPrompt], Scope | None]:
    def approve(prompt: ApprovalPrompt) -> Scope | None:
        c = prompt.classification
        allowed = c.category is Category.TEST_LINT and not c.needs_shell
        emit({
            "type": "approval",
            "command": c.command,
            "category": str(c.category),
            "allowed": allowed,
            "reason": "test/lint commands run automatically" if allowed
            else "only test and lint commands may run in the playground",
        })  # fmt: skip
        return Scope.ONCE if allowed else None

    return approve


def event_dict(event: AgentEvent) -> dict | None:
    """The browser's view of an agent event. Kept small: previews, not whole tool outputs."""
    match event:
        case Status(message=m):
            return {"type": "status", "message": m}
        case RetrievalDone(bundle=b):
            return {"type": "retrieval", "files": b.files, "summary": b.summary()}
        case AssistantText(text=t):
            return {"type": "text", "text": t}
        case EditProposed(block=b):
            return {"type": "edit", "path": b.path, "new_file": not b.search}
        case ToolStarted(call=c):
            return {"type": "tool", "id": c.id, "name": c.name, "args": c.arguments}
        case ToolFinished(call=c, is_error=e, summary=s):
            return {"type": "tool_done", "id": c.id, "name": c.name, "error": e, "preview": s}
        case EditsRejected(feedback=f, attempt=a):
            return {"type": "rejected", "attempt": a, "feedback": f[:1500]}
        case ValidationStarted(files=f, attempt=a):
            return {"type": "validating", "files": f, "attempt": a}
        case ValidationDone(report=r, attempt=a):
            return {"type": "validated", "attempt": a, "ok": r.ok, "summary": r.summary()}
    return None


def explanation(answer: str) -> str:
    """The model's prose without its edit blocks."""
    return re.sub(r"\n{3,}", "\n\n", _EDIT_BLOCK.sub("", answer)).strip()


def _git_init(root: Path) -> None:
    ident = ["-c", "user.name=playground", "-c", "user.email=playground@localhost",
             "-c", "commit.gpgsign=false"]  # fmt: skip
    run_git(root, "init", "-q")
    run_git(root, "add", "-A")
    run_git(root, *ident, "commit", "-q", "-m", "sample")


def _rmtree(path: Path) -> None:
    def force(func, p, _exc):  # git marks object files read-only on Windows
        Path(p).chmod(0o700)
        func(p)

    shutil.rmtree(path, onexc=force)


@dataclass
class RunLimits:
    max_usd: float
    max_seconds: int
    max_tool_calls: int


def run_task(
    sample: Sample,
    task: str,
    *,
    cfg: AgentConfig,
    providers: dict[str, Provider],
    embedder: Embedder | None,
    limits: RunLimits,
    cancel: CancelToken,
    emit: Emit,
) -> float | None:
    """Run one task and stream its events through `emit`. Returns the measured cost in USD
    (None if unknown). Ends with exactly one `done` or `error` event."""
    started = time.monotonic()
    cfg = cfg.model_copy(deep=True)
    cfg.budgets.max_usd = limits.max_usd
    cfg.budgets.max_seconds = limits.max_seconds
    cfg.budgets.max_tool_calls = limits.max_tool_calls
    cfg.validation.python = Path(sys.executable)  # the server's interpreter has pytest

    tmp = Path(tempfile.mkdtemp(prefix="playground-"))
    session: AgentSession | None = None
    gateway = build_gateway(cfg.llm, providers=providers)
    try:
        root = tmp / sample.id
        shutil.copytree(sample.path, root, ignore=shutil.ignore_patterns("__pycache__"))
        _git_init(root)
        session = AgentSession(
            Workspace(root=root, is_git=True), cfg, embedder,
            gateway=gateway, approver=playground_approver(emit),
        )  # fmt: skip
        problem = session.check_generation()
        if problem:
            emit({"type": "error", "message": f"model unavailable: {problem}"})
            return gateway.total_cost_usd
        emit({"type": "status", "message": "indexing the repository"})
        session.refresh_index()

        def on_event(event: AgentEvent) -> None:
            data = event_dict(event)
            if data is not None:
                emit(data)

        _, result = session.run_task(task, cancel, on_event)
        usage = result.usage
        emit({
            "type": "done",
            "status": result.status.value,
            "message": result.message,
            "answer": result.answer,
            "explanation": explanation(result.answer),
            "diff": plan_diff(result.plan) if result.plan else "",
            "files_changed": sorted(result.plan.changes) if result.plan else [],
            "validation": validation_dict(result.validation),
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "llm_calls": usage.llm_calls,
                "tool_calls": usage.tool_calls,
                "cost_usd": usage.cost_usd,
                "seconds": round(time.monotonic() - started, 1),
            },
        })  # fmt: skip
        return gateway.total_cost_usd
    except Exception as exc:  # the visitor gets a clean error; details stay in the server log
        emit({"type": "error", "message": f"the run failed: {type(exc).__name__}"})
        raise
    finally:
        if session is not None:
            session.close()
        _rmtree(tmp)
