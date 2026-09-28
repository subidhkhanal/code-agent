"""Audit log of every tool call and every user action on the workspace.

Stored per call: tool, actor (model | user | system), *redacted* arguments, approval id,
idempotency key, a SHA-256 of the output (never the output itself), whether redaction was
applied, status and timestamp. `.agent/` is a protected path, so the agent's tools cannot read
or rewrite this log.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from code_agent.db import utcnow
from code_agent.hashing import sha256_text
from code_agent.security.secrets import redact


@dataclass(frozen=True)
class AuditEntry:
    tool_call_id: str
    request_id: str | None
    approval_id: str | None
    tool_name: str
    actor: str
    redacted_args: str
    output_hash: str | None
    redaction_applied: bool
    status: str | None
    executed_at: str


class AuditLog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def record(
        self,
        *,
        tool_name: str,
        actor: str,
        args: dict[str, Any],
        request_id: str | None = None,
        tool_call_id: str | None = None,
        approval_id: str | None = None,
        output: str | None = None,
        status: str = "ok",
        idempotency_key: str | None = None,
    ) -> str:
        clean_args, kinds = redact(json.dumps(args, sort_keys=True, default=str))
        call_id = f"{tool_call_id or 'call'}-{uuid.uuid4().hex[:12]}"
        with self.conn:
            self.conn.execute(
                """INSERT INTO tool_audit_log(tool_call_id, request_id, approval_id, tool_name,
                       actor, redacted_args, output_hash, redaction_applied, idempotency_key,
                       status, executed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    call_id, request_id, approval_id, tool_name, actor, clean_args,
                    None if output is None else sha256_text(output), int(bool(kinds)),
                    idempotency_key, status, utcnow(),
                ),
            )  # fmt: skip
        return call_id

    def completed(self, idempotency_key: str) -> AuditEntry | None:
        row = self.conn.execute(
            f"SELECT {', '.join(AuditEntry.__dataclass_fields__)} FROM tool_audit_log "
            "WHERE idempotency_key = ? AND status = 'ok'",
            (idempotency_key,),
        ).fetchone()
        return None if row is None else AuditEntry(*row)

    def for_task(self, request_id: str) -> list[AuditEntry]:
        rows = self.conn.execute(
            f"SELECT {', '.join(AuditEntry.__dataclass_fields__)} FROM tool_audit_log "
            "WHERE request_id = ? ORDER BY rowid",
            (request_id,),
        ).fetchall()
        return [AuditEntry(*r) for r in rows]

    def latest_request_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT request_id FROM tool_audit_log WHERE request_id IS NOT NULL "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return None if row is None else row[0]
