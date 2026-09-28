"""Applying change sets to the workspace, and undoing them.

Every applied change set is recorded with the full before/after bytes of each file, so undo does
not depend on git and works for uncommitted and brand-new files too.

The record is written *before* the files are replaced (status PROPOSED), then flipped to APPLIED.
If the process dies mid-write, the PROPOSED record still lets `undo` restore whatever subset of
files was actually replaced.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from code_agent.db import utcnow
from code_agent.edits.apply import ApplyPlan
from code_agent.edits.atomic import FileWrite, write_all
from code_agent.hashing import sha256_bytes
from code_agent.security.paths import resolve_in_workspace


class ChangeSetStatus(StrEnum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    UNDONE = "UNDONE"


class UndoConflictError(RuntimeError):
    def __init__(self, paths: list[str]) -> None:
        super().__init__(
            "these files were modified after the change was applied, so undo would discard "
            "your edits: " + ", ".join(paths)
        )
        self.paths = paths


@dataclass(frozen=True)
class UndoResult:
    change_set_id: str
    restored: list[str]
    already_original: list[str]


class ChangeSetStore:
    def __init__(self, conn: sqlite3.Connection, root: Path) -> None:
        self.conn = conn
        self.root = root.resolve()

    def apply(self, plan: ApplyPlan, *, request_id: str | None = None) -> str:
        """Write a validated plan to disk. Returns the change-set id (for undo)."""
        if not plan.ok:
            raise ValueError("refusing to apply a plan with rejected blocks")
        change_set_id = uuid.uuid4().hex
        now = utcnow()
        with self.conn:
            self.conn.execute(
                "INSERT INTO change_sets(id, request_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (change_set_id, request_id, ChangeSetStatus.PROPOSED, now, now),
            )
            self.conn.executemany(
                """INSERT INTO change_set_files(change_set_id, file_path, before_hash, after_hash,
                       before_content, after_content, before_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        change_set_id,
                        c.rel_path,
                        c.before_hash,
                        c.after_hash,
                        c.before,
                        c.after,
                        c.mode,
                    )
                    for c in plan.changes.values()
                ],
            )
        writes = [
            FileWrite(c.abs_path, c.before_hash, c.after, c.mode) for c in plan.changes.values()
        ]
        try:
            write_all(writes)
        except BaseException:
            self._set_status(change_set_id, ChangeSetStatus.REJECTED)
            raise
        self._set_status(change_set_id, ChangeSetStatus.APPLIED)
        return change_set_id

    def latest_undoable(self) -> str | None:
        row = self.conn.execute(
            "SELECT id FROM change_sets WHERE status IN (?, ?) ORDER BY rowid DESC LIMIT 1",
            (ChangeSetStatus.APPLIED, ChangeSetStatus.PROPOSED),
        ).fetchone()
        return None if row is None else row[0]

    def undo(self, change_set_id: str | None = None) -> UndoResult:
        change_set_id = change_set_id or self.latest_undoable()
        if change_set_id is None:
            raise LookupError("nothing to undo")
        status = self.conn.execute(
            "SELECT status FROM change_sets WHERE id = ?", (change_set_id,)
        ).fetchone()
        if status is None or status[0] not in (ChangeSetStatus.APPLIED, ChangeSetStatus.PROPOSED):
            raise LookupError(f"change set {change_set_id} is not applied")

        rows = self.conn.execute(
            """SELECT file_path, before_hash, after_hash, before_content, before_mode
               FROM change_set_files WHERE change_set_id = ? ORDER BY file_path""",
            (change_set_id,),
        ).fetchall()
        writes: list[FileWrite] = []
        restored: list[str] = []
        untouched: list[str] = []
        conflicts: list[str] = []
        for file_path, before_hash, after_hash, before_content, before_mode in rows:
            path = resolve_in_workspace(self.root, file_path)
            current = path.read_bytes() if path.exists() else None
            current_hash = None if current is None else sha256_bytes(current)
            if current_hash == before_hash:
                untouched.append(file_path)  # never got written (crash) or already reverted
            elif current_hash == after_hash:
                content = None if before_content is None else bytes(before_content)
                writes.append(FileWrite(path, after_hash, content, before_mode))
                restored.append(file_path)
            else:
                conflicts.append(file_path)
        if conflicts:
            raise UndoConflictError(conflicts)
        write_all(writes)
        self._set_status(change_set_id, ChangeSetStatus.UNDONE)
        return UndoResult(change_set_id, restored, untouched)

    def _set_status(self, change_set_id: str, status: ChangeSetStatus) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE change_sets SET status = ?, updated_at = ? WHERE id = ?",
                (status, utcnow(), change_set_id),
            )
