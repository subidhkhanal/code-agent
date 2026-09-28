"""Unified diffs of a planned change set, for human review."""

from __future__ import annotations

import difflib

from code_agent.edits.apply import ApplyPlan, FileChange


def file_diff(change: FileChange, context_lines: int = 3) -> str:
    before = (change.before or b"").decode("utf-8", "replace").splitlines(keepends=True)
    after = change.after.decode("utf-8", "replace").splitlines(keepends=True)
    diff = difflib.unified_diff(
        before,
        after,
        fromfile="/dev/null" if change.before is None else f"a/{change.rel_path}",
        tofile=f"b/{change.rel_path}",
        n=context_lines,
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in diff)


def plan_diff(plan: ApplyPlan) -> str:
    return "".join(file_diff(plan.changes[path]) for path in sorted(plan.changes))


def diff_stats(plan: ApplyPlan) -> tuple[int, int]:
    """(lines added, lines removed) across the plan."""
    added = removed = 0
    for line in plan_diff(plan).splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
