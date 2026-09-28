"""Classify terminal commands the model wants to run. Pure code: the model never classifies.

Commands are never run through a shell. They are split with POSIX `shlex` rules and executed
as an argument vector, so there is no expansion, no redirection and no chaining unless a
command is *privileged* and the user explicitly approves running it through a shell.

Categories (ADR 0007):
  READ_ONLY   inspection commands; allowed after one approval per session
  TEST_LINT   test runners and linters in check mode; approval once or per session.
              Tests still execute repository code, which is why this is its own category.
  PRIVILEGED  everything else: mutating, networked, interpreters, unknown programs, shell
              syntax, paths outside the workspace or into sensitive files. Asks every time;
              a session-wide approval is impossible (enforced in approvals.py).

The lists are allowlists: a program not named here is PRIVILEGED. New tools default to "ask".
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath

from code_agent.security.paths import (
    PathOutsideWorkspaceError,
    SensitivePathPolicy,
    resolve_in_workspace,
    to_workspace_relpath,
)


class Category(StrEnum):
    READ_ONLY = "read-only"
    TEST_LINT = "test/lint"
    PRIVILEGED = "privileged"


@dataclass(frozen=True)
class Classification:
    command: str
    category: Category
    argv: tuple[str, ...]  # empty when the command could not be parsed
    cwd: str  # workspace-relative, "." for the root
    reasons: tuple[str, ...] = field(default=())
    needs_shell: bool = False  # uses pipes/redirects/chaining; only runnable via a shell

    @property
    def summary(self) -> str:
        why = f" ({'; '.join(self.reasons)})" if self.reasons else ""
        return f"{self.category}{why}"


# Anything that makes a shell do more than run one program with literal arguments.
_SHELL_SYNTAX = re.compile(r"[|;&<>`\n]|\$\(|\$\{|\$[A-Za-z_]|^\s*\(|\)\s*$")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

READ_ONLY_PROGRAMS = frozenset({
    "ls", "dir", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "pwd",
    "echo", "tree", "file", "stat", "which", "where", "sort", "uniq", "diff", "du", "realpath",
    "basename", "dirname", "type",
})  # fmt: skip
GIT_READ_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "grep", "describe",
    "shortlog", "ls-tree", "cat-file",
})  # fmt: skip
TEST_LINT_PROGRAMS = frozenset({"pytest", "py.test", "pyright", "mypy", "ruff", "unittest"})

# Flags that turn an otherwise read-only program into one that executes or writes.
_DANGEROUS_FLAGS: dict[str, tuple[str, ...]] = {
    "find": ("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0",
             "-fprintf", "-fls"),
    "rg": ("--pre", "--pre-glob"),
    "sort": ("-o", "--output"),
    "tree": ("-o",),
}  # fmt: skip
_GIT_DANGEROUS = ("-c", "--exec-path", "--git-dir", "--work-tree", "--output", "--ext-diff")


def split_command(command: str) -> list[str]:
    """POSIX quoting rules everywhere, except that on Windows a backslash is a path separator
    (`C:\\Python312\\python.exe`), not an escape character."""
    if os.name == "nt":
        command = command.replace("\\", "\\\\")
    return shlex.split(command, posix=True)


def _program(argv: list[str]) -> tuple[str, list[str]]:
    """Normalize argv[0] and unwrap `python -m <module>` into the module's name."""
    name = PurePosixPath(argv[0].replace("\\", "/")).name.lower().removesuffix(".exe")
    rest = argv[1:]
    if re.fullmatch(r"python(3(\.\d+)?)?|py", name) and len(rest) >= 2 and rest[0] == "-m":
        return rest[1].lower(), rest[2:]
    return name, rest


def _path_problems(
    args: list[str], cwd: Path, root: Path, sensitive: SensitivePathPolicy
) -> list[str]:
    problems = []
    for arg in args:
        if arg.startswith("-") or not arg:
            continue
        value = arg.split("=", 1)[1] if arg.startswith("--") and "=" in arg else arg
        if not re.search(r"[/\\.]", value) and not (cwd / value).exists():
            continue  # a plain word (search pattern, subcommand), not a path
        try:
            real = resolve_in_workspace(root, cwd / value)
        except PathOutsideWorkspaceError:
            problems.append(f"{value!r} is outside the workspace")
            continue
        if real != root and sensitive.is_sensitive(to_workspace_relpath(root, real)):
            problems.append(f"{value!r} is a protected path")
    return problems


def classify(
    command: str,
    *,
    root: Path,
    cwd: str = ".",
    sensitive: SensitivePathPolicy | None = None,
) -> Classification:
    sensitive = sensitive or SensitivePathPolicy()
    root = root.resolve()

    def privileged(
        *reasons: str, argv: tuple[str, ...] = (), shell: bool = False
    ) -> Classification:
        return Classification(command, Category.PRIVILEGED, argv, cwd, tuple(reasons), shell)

    try:
        real_cwd = resolve_in_workspace(root, cwd)
    except PathOutsideWorkspaceError:
        return privileged("working directory is outside the workspace")
    if not command.strip():
        return privileged("empty command")
    if _SHELL_SYNTAX.search(command):
        return privileged("uses shell syntax (pipes, redirects, chaining or expansion)",
                          shell=True)  # fmt: skip
    try:
        argv = split_command(command)
    except ValueError as exc:
        return privileged(f"cannot be parsed: {exc}")
    if not argv:
        return privileged("empty command")
    if _ENV_ASSIGNMENT.match(argv[0]):
        return privileged("sets environment variables", argv=tuple(argv))

    program, args = _program(argv)
    path_problems = _path_problems(args, real_cwd, root, sensitive)
    if path_problems:
        return privileged(*path_problems, argv=tuple(argv))

    rel_cwd = "." if real_cwd == root else to_workspace_relpath(root, real_cwd)
    ok = lambda category, *why: Classification(command, category, tuple(argv), rel_cwd, why)  # noqa: E731

    if program == "git":
        if any(a.split("=", 1)[0] in _GIT_DANGEROUS for a in args):
            return privileged("git option that can run programs or write files", argv=tuple(argv))
        sub = next((a for a in args if not a.startswith("-")), None)
        if sub in GIT_READ_SUBCOMMANDS:
            return ok(Category.READ_ONLY)
        if sub in ("branch", "remote", "tag", "stash") and all(
            a in (sub, "-a", "-r", "-v", "-vv", "--list", "-l") for a in args
        ):
            return ok(Category.READ_ONLY, f"lists {sub}")
        return privileged(f"git {sub or ''} can change the repository".strip(), argv=tuple(argv))

    if program in READ_ONLY_PROGRAMS:
        bad = [a for a in args if a.split("=", 1)[0] in _DANGEROUS_FLAGS.get(program, ())]
        if bad:
            return privileged(f"{program} {bad[0]} can execute or write", argv=tuple(argv))
        return ok(Category.READ_ONLY)

    if program in TEST_LINT_PROGRAMS:
        if program == "ruff":
            sub = next((a for a in args if not a.startswith("-")), "check")
            if sub == "format" and not {"--check", "--diff"} & set(args):
                return privileged("ruff format rewrites files", argv=tuple(argv))
            if sub == "check" and {"--fix", "--unsafe-fixes"} & set(args):
                return privileged("ruff check --fix rewrites files", argv=tuple(argv))
            if sub not in ("check", "format"):
                return privileged(f"ruff {sub} is not a check", argv=tuple(argv))
        if program == "ruff":
            return ok(Category.TEST_LINT)
        return ok(Category.TEST_LINT, "runs repository code")

    return privileged(f"{program!r} is not on the allowlist", argv=tuple(argv))
