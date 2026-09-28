"""Approvals, the command tool, idempotency and the audit log, end to end through the registry.

Commands used here are the test's own fixed commands, not model output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from code_agent.agent.tasks import TaskStatus, TaskStore, TaskUsage
from code_agent.agent.tools import Capability, ToolContext, ToolRegistry
from code_agent.config import BudgetConfig
from code_agent.llm.types import CancelToken, ToolCall
from code_agent.security.approvals import ApprovalManager, ApprovalPrompt, Scope
from code_agent.security.audit import AuditLog
from code_agent.security.commands import Category, classify
from code_agent.security.paths import SensitivePathPolicy

from .conftest import IndexedRepo

PY = Path(sys.executable).as_posix()


class ScriptedApprover:
    """Plays the user: answers prompts in order and records what it was asked."""

    def __init__(self, *answers: Scope | None, on_prompt=None) -> None:
        self.answers = list(answers)
        self.prompts: list[ApprovalPrompt] = []
        self.on_prompt = on_prompt

    def __call__(self, prompt: ApprovalPrompt) -> Scope | None:
        self.prompts.append(prompt)
        if self.on_prompt:
            self.on_prompt(prompt)
        return self.answers.pop(0)


class Env:
    def __init__(self, indexed: IndexedRepo, approver: ScriptedApprover) -> None:
        conn = indexed.store.conn
        self.request_id = TaskStore(conn, indexed.workspace.repo_id).create("t", BudgetConfig())
        self.approver = approver
        self.approvals = ApprovalManager(conn, approver)
        self.audit = AuditLog(conn)
        self.cancel = CancelToken()
        self.ctx = ToolContext(
            indexed.root, SensitivePathPolicy(), indexed.searcher, conn,
            indexed.workspace.repo_id, request_id=self.request_id, audit=self.audit,
            approvals=self.approvals, cancel=self.cancel,
        )  # fmt: skip
        self.registry = ToolRegistry(
            self.ctx, allowed=frozenset({Capability.READ, Capability.EXECUTE})
        )
        self.root = indexed.root
        self._n = 0

    def run(self, command: str, call_id: str | None = None, **extra):
        self._n += 1
        call = ToolCall(call_id or f"c{self._n}", "run_terminal_command",
                        {"command": command, **extra})  # fmt: skip
        return self.registry.execute(call)


def touch_cmd(path: Path) -> str:
    # A privileged command ('python' is not allowlisted) with an observable side effect.
    return f"{PY} -c \"open(r'{path.as_posix()}', 'a').write('x')\""


def test_read_only_session_grant_asks_once(indexed: IndexedRepo):
    env = Env(indexed, ScriptedApprover(Scope.SESSION))
    first = env.run("git status")
    second = env.run("git log --oneline -1")
    assert not first.is_error and not second.is_error, (first.content, second.content)
    assert len(env.approver.prompts) == 1
    assert env.approver.prompts[0].allowed_scopes == (Scope.ONCE, Scope.SESSION)


def test_test_lint_grant_does_not_cover_privileged(indexed: IndexedRepo):
    marker = indexed.root / "ran.txt"
    env = Env(indexed, ScriptedApprover(Scope.SESSION, None))
    env.run("ruff --version")  # test/lint ('ruff' allowlisted), granted for the session
    result = env.run(touch_cmd(marker))
    assert result.is_error and "denied" in result.content
    assert not marker.exists()
    assert len(env.approver.prompts) == 2  # privileged still asked


def test_privileged_is_asked_every_time_and_session_is_downgraded(indexed: IndexedRepo):
    marker = indexed.root / "count.txt"
    env = Env(indexed, ScriptedApprover(Scope.SESSION, Scope.ONCE))
    assert not env.run(touch_cmd(marker)).is_error
    assert not env.run(touch_cmd(marker)).is_error
    assert marker.read_text() == "xx"
    assert len(env.approver.prompts) == 2
    assert all(p.allowed_scopes == (Scope.ONCE,) for p in env.approver.prompts)
    scopes = [r[0] for r in indexed.store.conn.execute("SELECT scope FROM approvals")]
    assert scopes == ["once", "once"]  # the UI's "session" answer was not honored


def test_denied_command_never_runs(indexed: IndexedRepo):
    marker = indexed.root / "should-not-exist.txt"
    env = Env(indexed, ScriptedApprover(None))
    result = env.run(touch_cmd(marker))
    assert result.is_error and "the user denied" in result.content
    assert not marker.exists()


def test_cancel_while_approval_prompt_is_open_blocks_execution(indexed: IndexedRepo):
    marker = indexed.root / "cancelled.txt"
    env: Env
    approver = ScriptedApprover(Scope.ONCE, on_prompt=lambda _p: env.cancel.cancel())
    env = Env(indexed, approver)
    result = env.run(touch_cmd(marker))
    assert result.is_error and "cancelled" in result.content
    assert not marker.exists()


def test_cancelled_task_status_blocks_execution(indexed: IndexedRepo):
    marker = indexed.root / "cancelled2.txt"
    env = Env(indexed, ScriptedApprover(Scope.ONCE))
    TaskStore(indexed.store.conn, indexed.workspace.repo_id).update(
        env.request_id, TaskStatus.CANCELLED, TaskUsage()
    )
    assert "cancelled" in env.run(touch_cmd(marker)).content
    assert not marker.exists()


def test_one_time_approval_cannot_be_reused(indexed: IndexedRepo):
    env = Env(indexed, ScriptedApprover(Scope.ONCE))
    c = classify(touch_cmd(indexed.root / "x.txt"), root=indexed.root)
    approval = env.approvals.authorize(env.request_id, c)
    assert approval is not None
    assert env.approvals.recheck(approval, env.request_id, env.cancel) is None
    assert "already used" in env.approvals.recheck(approval, env.request_id, env.cancel)


def test_revoked_session_grant_fails_recheck(indexed: IndexedRepo):
    env = Env(indexed, ScriptedApprover(Scope.SESSION))
    approval = env.approvals.authorize(env.request_id, classify("git status", root=indexed.root))
    assert approval is not None and approval.category is Category.READ_ONLY
    env.approvals.revoke_session_grants()
    assert "revoked" in env.approvals.recheck(approval, env.request_id, env.cancel)


def test_same_tool_call_is_not_executed_twice(indexed: IndexedRepo):
    marker = indexed.root / "idempotent.txt"
    env = Env(indexed, ScriptedApprover(Scope.ONCE, Scope.ONCE))
    env.run(touch_cmd(marker), call_id="same-call")
    second = env.run(touch_cmd(marker), call_id="same-call")  # e.g. a retry after a crash
    assert marker.read_text() == "x"
    assert "not run again" in second.content
    assert len(env.approver.prompts) == 1  # not even asked again


def test_shell_command_runs_only_as_an_explicitly_approved_privileged_command(indexed):
    env = Env(indexed, ScriptedApprover(Scope.ONCE))
    result = env.run("git status && git log --oneline -1")
    prompt = env.approver.prompts[0]
    assert prompt.classification.category is Category.PRIVILEGED
    assert prompt.classification.needs_shell and not result.is_error, result.content


def test_missing_program_is_a_clean_error(indexed: IndexedRepo):
    env = Env(indexed, ScriptedApprover(Scope.ONCE))
    result = env.run("definitely-not-a-real-program-xyz --help")
    assert result.is_error and "could not start" in result.content


def test_every_call_is_audited_with_redacted_args(indexed: IndexedRepo):
    token = "ghp_" + "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2Ji1Hg0Fe9Dc8Ba7"
    env = Env(indexed, ScriptedApprover(None, Scope.ONCE))
    env.run(f"git log --grep {token}")  # denied
    env.run("git status")  # approved once
    env.registry.execute(ToolCall("r1", "read_file", {"path": "auth/tokens.py"}))
    env.registry.execute(ToolCall("r2", "read_file", {"path": ".env"}))  # rejected by policy
    entries = env.audit.for_task(env.request_id)
    assert [(e.tool_name, e.status) for e in entries] == [
        ("run_terminal_command", "denied"),
        ("run_terminal_command", "ok"),
        ("read_file", "ok"),
        ("read_file", "error"),
    ]
    assert token not in entries[0].redacted_args and entries[0].redaction_applied
    assert json.loads(entries[0].redacted_args)["command"].startswith("git log --grep [REDACTED")
    assert entries[1].approval_id is not None and entries[1].output_hash is not None
    assert all(e.actor == "model" for e in entries)


@pytest.mark.parametrize("scope", [Scope.ONCE, Scope.SESSION])
def test_execution_disabled_without_approval_manager(indexed: IndexedRepo, scope: Scope):
    env = Env(indexed, ScriptedApprover(scope))
    env.ctx.approvals = None
    result = env.run("git status")
    assert result.is_error and "not enabled" in result.content


@pytest.mark.parametrize("command", ["pytest -q tests/test_tokens.py", "python -m pytest -q tests"])
def test_test_commands_run_via_a_known_interpreter_not_path(
    indexed: IndexedRepo, monkeypatch, command: str
):
    # Found in a real-model run: a bare `pytest` failed because it was not on PATH. PATH here
    # holds only system dirs, so success proves the command did not rely on PATH lookup.
    import os

    system_dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep)
                   if "system32" in d.lower() or d in ("/usr/bin", "/bin")]  # fmt: skip
    monkeypatch.setenv("PATH", os.pathsep.join(system_dirs))
    env = Env(indexed, ScriptedApprover(Scope.ONCE))
    result = env.run(command)
    assert not result.is_error, result.content
    assert "1 failed, 1 passed" in result.content  # the fixture's planted bug, as expected
