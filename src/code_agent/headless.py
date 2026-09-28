"""Headless autonomous mode: one task, no human, inside a container (ADR 0009).

`agent run --task ... --headless --auto-approve` runs the normal engine with an approver that
grants every command *once* (so each one is still individually approved and audited), applies
the resulting change set to the workspace, and writes:

  <out>/patch.diff     `git diff` of the workspace after the task (the deliverable)
  <out>/report.json    status, usage, attempts, validation results (schema: HeadlessReport)

Auto-approval is only safe where the OS provides isolation, so this refuses to start unless it
detects a container. `allow_outside_container` exists for tests and must be passed explicitly.

This is also the integration point for an external orchestrator (`spawn_sandbox`): run the
image with a task and a mounted repo, read back `report.json` and `patch.diff`. See
docs/headless.md for the contract.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from code_agent.agent.loop import AgentEvent
from code_agent.agent.session import AgentSession
from code_agent.agent.tasks import TaskStatus
from code_agent.config import AgentConfig
from code_agent.index.embeddings import Embedder
from code_agent.llm.gateway import Gateway
from code_agent.security.approvals import ApprovalPrompt, Scope
from code_agent.workspace import Workspace, run_git

REPORT_VERSION = 1


class HeadlessRefusedError(RuntimeError):
    pass


def in_container() -> bool:
    """Filesystem markers only: an environment variable could be set by anyone, anywhere."""
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "containerd", "kubepods", "libpod"))


def auto_approve(prompt: ApprovalPrompt) -> Scope | None:
    # One-time approval per command, so every execution still has its own approval record.
    return Scope.ONCE


@dataclass
class HeadlessReport:
    version: int
    status: str  # SUCCEEDED | FAILED | CANCELLED
    applied: bool  # whether edits were written to the workspace
    message: str
    task: str
    request_id: str
    routes: dict[str, list[str]]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    llm_calls: int
    tool_calls: int
    edit_attempts: int
    fix_attempts: int
    rejections: list[str]
    files_changed: list[str]
    validation: dict | None
    seconds: float
    events: list[str] = field(default_factory=list)  # compact trace for debugging failures


def run_headless(
    workspace: Workspace,
    cfg: AgentConfig,
    task: str,
    *,
    out_dir: Path,
    embedder: Embedder | None,
    gateway: Gateway | None = None,
    allow_outside_container: bool = False,
) -> HeadlessReport:
    if not in_container() and not allow_outside_container:
        raise HeadlessRefusedError(
            "headless auto-approve mode only runs inside a container; "
            "use `agent chat` for interactive work"
        )
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    events: list[str] = []

    def trace(event: AgentEvent) -> None:
        events.append(f"{type(event).__name__}: {_describe(event)}"[:300])

    session = AgentSession(workspace, cfg, embedder, gateway=gateway, approver=auto_approve)
    try:
        problem = session.check_generation()
        if problem:
            raise HeadlessRefusedError(f"generation unavailable: {problem}")
        session.refresh_index()
        request_id, result = session.run_task(task, on_event=trace)
        applied = False
        if result.plan is not None and result.plan.changes:
            session.apply(request_id, result)
            applied = True
            new_files = [p for p, c in result.plan.changes.items() if c.before is None]
            if new_files:
                # Plain `git diff` omits untracked files; intent-to-add makes new files show up
                # in the patch without staging their content.
                run_git(workspace.root, "add", "--intent-to-add", "--", *new_files)
        elif result.status is TaskStatus.SUCCEEDED:
            session.tasks.update(request_id, TaskStatus.SUCCEEDED, result.usage)
    finally:
        session.close()

    diff = run_git(workspace.root, "diff", "--no-color", "--no-ext-diff")
    (out_dir / "patch.diff").write_bytes(diff.stdout)
    report = HeadlessReport(
        version=REPORT_VERSION,
        status=result.status.value,
        applied=applied,
        message=result.message,
        task=task,
        request_id=request_id,
        routes={role: list(specs) for role, specs in cfg.llm.routes.items()},
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=result.usage.cost_usd,
        llm_calls=result.usage.llm_calls,
        tool_calls=result.usage.tool_calls,
        edit_attempts=result.usage.edit_attempts,
        fix_attempts=result.usage.fix_attempts,
        rejections=result.rejections,
        files_changed=sorted(result.plan.changes) if result.plan else [],
        validation=_validation_dict(result.validation),
        seconds=round(time.monotonic() - started, 1),
        events=events,
    )
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def _validation_dict(report) -> dict | None:
    if report is None:
        return None
    return {
        "ok": report.ok,
        "summary": report.summary(),
        "attempted": report.attempted,
        "skipped": report.skipped,
        "new_diagnostics": [d.render() for d in report.new_diagnostics],
        "regressions": report.regressions,
        "still_failing": report.still_failing,
        "fixed": report.fixed,
        "tests_run": report.tests_run,
    }


def _describe(event: AgentEvent) -> str:
    for attr in ("message", "text", "feedback"):
        if hasattr(event, attr):
            return str(getattr(event, attr)).strip()
    if hasattr(event, "call"):
        call = event.call  # type: ignore[attr-defined]
        return f"{call.name}({json.dumps(call.arguments)[:200]})"
    if hasattr(event, "block"):
        return event.block.path  # type: ignore[attr-defined]
    if hasattr(event, "bundle"):
        return event.bundle.summary()  # type: ignore[attr-defined]
    if hasattr(event, "report"):
        return event.report.summary()  # type: ignore[attr-defined]
    if hasattr(event, "files"):
        return ", ".join(event.files)  # type: ignore[attr-defined]
    return ""
