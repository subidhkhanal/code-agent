"""Approvals for terminal commands. Policy lives here, in code; the model can only *ask*.

Rules (ADR 0007):
* READ_ONLY   ask once; the user may grant it for the rest of the session.
* TEST_LINT   ask once or grant for the session (user's choice).
* PRIVILEGED  ask every time. A session grant is impossible: any requested or chosen
              "session" scope is downgraded to "once" here, whatever the model or UI asks for.

Every decision is stored with an `approval_id`. Right before a command runs, `recheck()`
verifies the approval is still valid (approved, not revoked, a one-time approval not already
used) and that the task has not been cancelled. A one-time approval is consumed atomically with
a conditional UPDATE, so it can authorize exactly one execution even if something retries.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from code_agent.db import utcnow
from code_agent.llm.types import CancelToken
from code_agent.security.commands import Category, Classification


class Scope(StrEnum):
    ONCE = "once"
    SESSION = "session"


@dataclass(frozen=True)
class ApprovalPrompt:
    """What the UI shows the user."""

    request_id: str
    classification: Classification
    allowed_scopes: tuple[Scope, ...]


@dataclass(frozen=True)
class Approval:
    approval_id: str
    command: str
    category: Category
    scope: Scope


# The UI returns the user's decision: None = denied, else the scope they granted.
Approver = Callable[[ApprovalPrompt], Scope | None]


def deny_all(_: ApprovalPrompt) -> Scope | None:
    return None


class ApprovalManager:
    def __init__(self, conn: sqlite3.Connection, approver: Approver = deny_all) -> None:
        self.conn = conn
        self.approver = approver
        self._session_grants: dict[Category, str] = {}  # category -> approval_id of the grant

    def authorize(self, request_id: str, classification: Classification) -> Approval | None:
        category = classification.category
        if category is not Category.PRIVILEGED and category in self._session_grants:
            return Approval(
                self._session_grants[category], classification.command, category, Scope.SESSION
            )

        allowed = (Scope.ONCE,) if category is Category.PRIVILEGED else (Scope.ONCE, Scope.SESSION)
        choice = self.approver(ApprovalPrompt(request_id, classification, allowed))
        if choice is not None and choice not in allowed:
            choice = Scope.ONCE  # never trust the UI (or anything else) to widen the scope
        approval_id = uuid.uuid4().hex
        with self.conn:
            self.conn.execute(
                """INSERT INTO approvals(approval_id, request_id, command, classification, scope,
                       decision, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    approval_id, request_id, classification.command, category,
                    choice or Scope.ONCE, "approved" if choice else "denied", utcnow(),
                ),
            )  # fmt: skip
        if choice is None:
            return None
        if choice is Scope.SESSION:
            self._session_grants[category] = approval_id
        return Approval(approval_id, classification.command, category, choice)

    def recheck(self, approval: Approval, request_id: str, cancel: CancelToken) -> str | None:
        """Final gate before execution. Returns None if OK to run, else the reason not to."""
        if cancel.cancelled:
            return "the task was cancelled"
        task = self.conn.execute(
            "SELECT status FROM agent_tasks WHERE request_id = ?", (request_id,)
        ).fetchone()
        if task is not None and task[0] == "CANCELLED":
            return "the task was cancelled"
        row = self.conn.execute(
            "SELECT decision, revoked_at, scope FROM approvals WHERE approval_id = ?",
            (approval.approval_id,),
        ).fetchone()
        if row is None or row[0] != "approved":
            return "no valid approval"
        if row[1] is not None:
            return "the approval was revoked"
        if row[2] == Scope.ONCE:
            with self.conn:
                consumed = self.conn.execute(
                    "UPDATE approvals SET consumed_at = ? "
                    "WHERE approval_id = ? AND consumed_at IS NULL",
                    (utcnow(), approval.approval_id),
                ).rowcount
            if consumed != 1:
                return "the one-time approval was already used"
        return None

    def revoke_session_grants(self) -> None:
        with self.conn:
            for approval_id in self._session_grants.values():
                self.conn.execute(
                    "UPDATE approvals SET revoked_at = ? WHERE approval_id = ?",
                    (utcnow(), approval_id),
                )
        self._session_grants.clear()
