"""Persistence for agent tasks (the `agent_tasks` table): status, budgets and usage."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum

from code_agent.config import BudgetConfig
from code_agent.db import utcnow


class TaskStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    edit_attempts: int = 0
    fix_attempts: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TaskStore:
    def __init__(self, conn: sqlite3.Connection, repo_id: str) -> None:
        self.conn = conn
        self.repo_id = repo_id

    def create(
        self,
        task: str,
        budgets: BudgetConfig,
        *,
        base_revision: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        request_id = uuid.uuid4().hex
        now = utcnow()
        with self.conn:
            self.conn.execute(
                """INSERT INTO agent_tasks(request_id, conversation_id, repo_id, base_revision,
                       task, status, budgets_json, usage_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id, conversation_id, self.repo_id, base_revision, task,
                    TaskStatus.RUNNING, budgets.model_dump_json(), json.dumps(asdict(TaskUsage())),
                    now, now,
                ),
            )  # fmt: skip
        return request_id

    def update(self, request_id: str, status: TaskStatus, usage: TaskUsage) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE agent_tasks SET status = ?, usage_json = ?, updated_at = ? "
                "WHERE request_id = ?",
                (status, json.dumps(asdict(usage)), utcnow(), request_id),
            )

    def status(self, request_id: str) -> TaskStatus | None:
        row = self.conn.execute(
            "SELECT status FROM agent_tasks WHERE request_id = ?", (request_id,)
        ).fetchone()
        return None if row is None else TaskStatus(row[0])
