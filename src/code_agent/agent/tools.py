"""Tools the model can call.

Every call goes through `ToolRegistry.execute`, which enforces, in code:
1. the tool exists and its *capability* is allowed in the current mode (the model cannot grant
   itself capabilities; the allowed set is fixed when the registry is built),
2. arguments validate against a strict schema (unknown keys, wrong types, out-of-range values
   are rejected; arguments are model output and therefore untrusted),
3. paths are canonicalized and must be inside the workspace and not sensitive,
4. output is truncated with a visible marker and wrapped as untrusted data.

Files whose content is shown to the model are recorded in `ToolContext.base_hashes`; Fast Apply
uses those hashes to reject edits against stale content (ADR 0005).
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import jedi
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from code_agent.hashing import sha256_bytes, sha256_text
from code_agent.index.files import read_source_text
from code_agent.llm.types import CancelToken, ToolCall, ToolSpec
from code_agent.retrieval.search import Searcher
from code_agent.security.approvals import ApprovalManager
from code_agent.security.audit import AuditLog
from code_agent.security.commands import classify
from code_agent.security.paths import (
    PathOutsideWorkspaceError,
    SensitivePathPolicy,
    resolve_in_workspace,
    to_workspace_relpath,
)
from code_agent.security.runner import run_command

MAX_OUTPUT_CHARS = 16_000
MAX_READ_LINES = 400
MAX_DEFINITION_LINES = 120


class Capability(StrEnum):
    READ = "read"  # read workspace files and the index
    EXECUTE = "execute"  # run commands (M3, behind approvals)


@dataclass
class ToolContext:
    root: Path
    sensitive: SensitivePathPolicy
    searcher: Searcher
    conn: sqlite3.Connection
    repo_id: str
    base_hashes: dict[str, str] = field(default_factory=dict)
    # Set for real tasks; command execution and auditing are disabled without them.
    request_id: str | None = None
    audit: AuditLog | None = None
    approvals: ApprovalManager | None = None
    cancel: CancelToken = field(default_factory=CancelToken)
    call_id: str = ""  # id of the tool call currently executing (set by the registry)


@dataclass(frozen=True)
class ToolOutput:
    """A tool result plus the approval/idempotency metadata the audit log needs."""

    content: str
    approval_id: str | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True)
class ToolResult:
    tool: str
    content: str
    is_error: bool = False
    truncated: bool = False

    def render(self) -> str:
        """What the model sees. The wrapper marks the content as data, not instructions."""
        status = "error" if self.is_error else "ok"
        return (
            f'<tool_output tool="{self.tool}" status="{status}" trust="untrusted">\n'
            f"{self.content}\n</tool_output>"
        )


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=2_000)


class ReadFileArgs(_Args):
    path: str = Field(description="Workspace-relative file path")
    start_line: int = Field(default=1, ge=1, description="First line to show (1-based)")
    end_line: int | None = Field(default=None, ge=1, description="Last line to show (inclusive)")


class SearchArgs(_Args):
    query: str = Field(min_length=1, max_length=500, description="Natural language or identifiers")
    k: int = Field(default=8, ge=1, le=20, description="Number of results")


class CommandArgs(_Args):
    command: str = Field(min_length=1, max_length=2_000, description="One command, no shell")
    cwd: str = Field(default=".", max_length=500, description="Workspace-relative directory")
    timeout_s: int = Field(default=120, ge=1, le=900)


class SymbolArgs(_Args):
    symbol: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$",
        description="Name or qualified name, e.g. verify_token or TokenStore.verify_token",
    )


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[_Args]
    capability: Capability
    run: Callable[[ToolContext, Any], str | ToolOutput]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.args.model_json_schema())


class ToolError(Exception):
    """Raised inside a tool for an expected failure; shown to the model as an error result."""


class ToolDeniedError(ToolError):
    """The user (or policy) refused the action."""


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n[... truncated {len(text) - limit:,} characters ...]", True


def _safe_path(ctx: ToolContext, raw: str) -> tuple[Path, str]:
    try:
        real = resolve_in_workspace(ctx.root, raw)
    except PathOutsideWorkspaceError as exc:
        raise ToolError(f"access denied: {exc}") from exc
    rel = to_workspace_relpath(ctx.root, real)
    if ctx.sensitive.is_sensitive(rel):
        raise ToolError(f"access denied: {rel} is a protected path")
    return real, rel


def _index_hashes(ctx: ToolContext, files: set[str]) -> None:
    for file_path in files:
        row = ctx.conn.execute(
            "SELECT content_hash FROM indexed_files WHERE repo_id = ? AND file_path = ?",
            (ctx.repo_id, file_path),
        ).fetchone()
        if row is not None:
            ctx.base_hashes.setdefault(file_path, row[0])


def gutter(lines: list[str], first_line: int) -> str:
    """`  12 | code`. The same format Fast Apply strips if the model copies it into SEARCH."""
    return "\n".join(f"{first_line + i:>5} | {line}" for i, line in enumerate(lines))


# -- tool implementations -----------------------------------------------------------------------


def read_file(ctx: ToolContext, args: ReadFileArgs) -> str:
    real, rel = _safe_path(ctx, args.path)
    if not real.is_file():
        raise ToolError(f"{rel} does not exist or is not a file")
    data = real.read_bytes()
    text = read_source_text(data)
    if text is None:
        raise ToolError(f"{rel} is binary or not UTF-8")
    ctx.base_hashes[rel] = sha256_bytes(data)  # the model has now seen this exact content
    lines = text.splitlines()
    end = min(args.end_line or len(lines), len(lines), args.start_line + MAX_READ_LINES - 1)
    if args.start_line > max(len(lines), 1):
        raise ToolError(f"{rel} has only {len(lines)} lines")
    shown = gutter(lines[args.start_line - 1 : end], args.start_line)
    more = ""
    if end < len(lines):
        more = (
            f"\n[lines {end + 1}-{len(lines)} not shown; call read_file with start_line={end + 1}]"
        )
    return f"{rel} (lines {args.start_line}-{end} of {len(lines)})\n{shown}{more}"


def search_codebase(ctx: ToolContext, args: SearchArgs) -> str:
    result = ctx.searcher.search(args.query, k=args.k)
    if not result.hits:
        return "No results." + ("\n" + "\n".join(result.notes) if result.notes else "")
    _index_hashes(ctx, {h.file_path for h in result.hits})
    out = []
    for i, hit in enumerate(result.hits, 1):
        preview = hit.content.split("\n")
        body = "\n".join(preview[:12]) + ("\n    ..." if len(preview) > 12 else "")
        label = f" {hit.symbol} ({hit.kind})" if hit.symbol else f" ({hit.kind})"
        out.append(f"{i}. {hit.file_path}:{hit.start_line}-{hit.end_line}{label}\n{body}")
    return "\n\n".join(out)


def _definitions(ctx: ToolContext, symbol: str) -> list[sqlite3.Row]:
    if "." in symbol:
        sql = """SELECT file_path, symbol, kind, start_line, end_line, content FROM file_chunks
                 WHERE repo_id = ? AND (symbol = ? OR substr(symbol, -length(?) - 1) = '.' || ?)
                 AND kind IN ('function', 'class', 'method') ORDER BY file_path"""
        params: tuple = (ctx.repo_id, symbol, symbol, symbol)
    else:
        sql = """SELECT file_path, symbol, kind, start_line, end_line, content FROM file_chunks
                 WHERE repo_id = ? AND name = ? AND kind IN ('function', 'class', 'method')
                 ORDER BY file_path"""
        params = (ctx.repo_id, symbol)
    rows = ctx.conn.execute(sql, params).fetchall()
    return [r for r in rows if not ctx.sensitive.is_sensitive(r[0])]


def get_definition(ctx: ToolContext, args: SymbolArgs) -> str:
    rows = _definitions(ctx, args.symbol)
    if not rows:
        raise ToolError(f"no definition of {args.symbol!r} in the index")
    _index_hashes(ctx, {r[0] for r in rows})
    out = []
    for file_path, symbol, kind, start, end, content in rows[:5]:
        lines = content.split("\n")
        if len(lines) > MAX_DEFINITION_LINES:
            lines = [*lines[:MAX_DEFINITION_LINES], "    ... (use read_file for the rest)"]
        out.append(f"{file_path}:{start}-{end} {symbol} ({kind})\n{gutter(lines, start)}")
    extra = f"\n\n[{len(rows) - 5} more definitions not shown]" if len(rows) > 5 else ""
    return "\n\n".join(out) + extra


_DEF_LINE = r"^(\s*(?:async\s+)?(?:def|class)\s+){name}\b"


def get_references(ctx: ToolContext, args: SymbolArgs) -> str:
    rows = _definitions(ctx, args.symbol)
    if not rows:
        raise ToolError(f"no definition of {args.symbol!r} in the index")
    name = args.symbol.rsplit(".", 1)[-1]
    file_path, _, _, start, _, content = rows[0]
    pattern = re.compile(_DEF_LINE.format(name=re.escape(name)))
    for offset, line in enumerate(content.split("\n")):
        if m := pattern.match(line):
            line_no, column = start + offset, len(m.group(1))
            break
    else:
        raise ToolError(f"could not locate the definition line of {args.symbol!r}")

    project = jedi.Project(ctx.root)
    script = jedi.Script(path=str(ctx.root / file_path), project=project)
    refs = script.get_references(line_no, column, scope="project")
    out = []
    for ref in refs:
        if ref.module_path is None or ref.line is None:
            continue
        try:
            rel = to_workspace_relpath(ctx.root, Path(ref.module_path).resolve())
        except ValueError:
            continue  # outside the workspace (stdlib, site-packages)
        if ctx.sensitive.is_sensitive(rel):
            continue
        kind = "definition" if ref.is_definition() else "reference"
        out.append(f"{rel}:{ref.line} [{kind}] {ref.get_line_code().strip()}")
    note = f" (showing definition in {file_path}; {len(rows)} candidates)" if len(rows) > 1 else ""
    return f"{len(out)} locations for {args.symbol}{note}:\n" + "\n".join(out)


def _shell_argv(command: str) -> list[str]:
    """Only used for commands the user explicitly approved *as shell commands*."""
    if os.name == "nt":
        return ["cmd", "/d", "/s", "/c", command]
    return ["/bin/sh", "-c", command]


def run_terminal_command(ctx: ToolContext, args: CommandArgs) -> ToolOutput:
    if ctx.approvals is None or ctx.request_id is None:
        raise ToolError("running commands is not enabled in this session")
    c = classify(args.command, root=ctx.root, cwd=args.cwd, sensitive=ctx.sensitive)
    # Same task + same tool call + same command = same key. If the loop retries a call that
    # already completed (e.g. after a crash), the command is not run a second time.
    key = sha256_text("\x00".join([ctx.request_id, ctx.call_id, args.command, c.cwd]))
    if ctx.audit is not None and ctx.audit.completed(key) is not None:
        # The key stays with the execution that owns it; the replay is audited without it.
        return ToolOutput("this exact command already ran for this tool call; not run again")
    approval = ctx.approvals.authorize(ctx.request_id, c)
    if approval is None:
        raise ToolDeniedError(f"the user denied running `{args.command}` ({c.summary})")
    reason = ctx.approvals.recheck(approval, ctx.request_id, ctx.cancel)
    if reason is not None:
        raise ToolDeniedError(f"`{args.command}` was not run: {reason}")
    argv = _shell_argv(args.command) if c.needs_shell else list(c.argv)
    cwd = resolve_in_workspace(ctx.root, args.cwd)
    try:
        result = run_command(argv, cwd, timeout_s=args.timeout_s, cancel=ctx.cancel)
    except OSError as exc:
        # e.g. a shell built-in such as Windows `dir`/`echo`, which has no executable to start
        raise ToolError(f"could not start `{argv[0]}`: {exc.strerror or exc}") from exc
    return ToolOutput(f"[{c.summary}]\n{result.render()}", approval.approval_id, key)


TOOLS: tuple[Tool, ...] = (
    Tool("read_file", "Read a workspace file (with line numbers in a gutter). Always read a file "
         "before editing it.", ReadFileArgs, Capability.READ, read_file),
    Tool("search_codebase", "Hybrid keyword + semantic search over the indexed workspace.",
         SearchArgs, Capability.READ, search_codebase),
    Tool("get_definition", "Show the definition(s) of a function, class or method.",
         SymbolArgs, Capability.READ, get_definition),
    Tool("get_references", "List where a function, class or method is used, across the project.",
         SymbolArgs, Capability.READ, get_references),
    Tool("run_terminal_command", "Run one command in the workspace (no shell: no pipes, "
         "redirects or chaining). Read-only and test/lint commands may be pre-approved; "
         "anything else asks the user every time.", CommandArgs, Capability.EXECUTE,
         run_terminal_command),
)  # fmt: skip


class ToolRegistry:
    def __init__(
        self,
        ctx: ToolContext,
        *,
        allowed: frozenset[Capability] = frozenset({Capability.READ}),
        tools: tuple[Tool, ...] = TOOLS,
    ) -> None:
        self.ctx = ctx
        self._allowed = allowed  # fixed at construction; nothing at runtime can widen it
        self._tools = {t.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values() if t.capability in self._allowed]

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate, run and audit one tool call. Every outcome is audited, including rejects."""
        output = ToolOutput("")
        status = "ok"
        tool = self._tools.get(call.name)
        if tool is None:
            result = ToolResult(call.name, f"unknown tool {call.name!r}", is_error=True)
            status = "unknown_tool"
        elif tool.capability not in self._allowed:
            result = ToolResult(call.name, f"tool {call.name!r} is not permitted", is_error=True)
            status = "not_permitted"
        else:
            try:
                args = tool.args.model_validate(call.arguments)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}"
                    for e in exc.errors()
                )
                result = ToolResult(call.name, f"invalid arguments: {problems}", is_error=True)
                status = "invalid_arguments"
            else:
                self.ctx.call_id = call.id
                try:
                    raw = tool.run(self.ctx, args)
                    output = raw if isinstance(raw, ToolOutput) else ToolOutput(raw)
                    text, truncated = _truncate(output.content)
                    result = ToolResult(call.name, text, truncated=truncated)
                except ToolDeniedError as exc:
                    result = ToolResult(call.name, str(exc), is_error=True)
                    status = "denied"
                except ToolError as exc:
                    result = ToolResult(call.name, str(exc), is_error=True)
                    status = "error"
        if self.ctx.audit is not None:
            self.ctx.audit.record(
                tool_name=call.name, actor="model", args=call.arguments,
                request_id=self.ctx.request_id, tool_call_id=call.id,
                approval_id=output.approval_id, output=result.content, status=status,
                idempotency_key=output.idempotency_key if status == "ok" else None,
            )  # fmt: skip
        return result
