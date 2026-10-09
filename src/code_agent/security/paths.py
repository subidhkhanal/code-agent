"""Path canonicalization and the sensitive-path policy.

Two separate questions are answered here:

1. Is this path inside the workspace? (`resolve_in_workspace`) Symlinks and `..` are resolved
   before the check, so `docs/../../etc/passwd` or a symlink pointing at `~/.ssh` is rejected.
2. Is this path sensitive? (`SensitivePathPolicy`) Sensitive files are never indexed, never read
   by a tool, never edited, and never sent to the LLM.

The default sensitive patterns are hard-coded. User config can add patterns but cannot remove or
negate the defaults, so no config file (and certainly no model output) can loosen the policy.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import pathspec

# gitignore-style patterns, matched case-insensitively against workspace-relative POSIX paths.
DEFAULT_SENSITIVE_PATTERNS: tuple[str, ...] = (
    # Environment / dotenv files
    ".env",
    ".env.*",
    "*.env",
    # Keys and certificates
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    # Credential stores and tool configs that commonly hold tokens
    ".ssh/",
    ".gnupg/",
    ".kube/",
    ".docker/config.json",
    ".netrc",
    ".pypirc",
    ".npmrc",
    ".git-credentials",
    "credentials.json",
    "service-account*.json",
    "*.tfstate",
    "*.tfstate.*",
    # Conventional secret directories
    "secrets/",
    ".secrets/",
    # The agent's own state (index, audit log, undo records) and git internals (hooks = code exec)
    ".agent/",
    ".git/",
)


class PathPolicyError(ValueError):
    """Base class for path-policy violations."""


class PathOutsideWorkspaceError(PathPolicyError):
    pass


class SensitivePathError(PathPolicyError):
    pass


class InvalidPatternError(PathPolicyError):
    pass


class SensitivePathPolicy:
    """Decides whether a workspace-relative path is sensitive."""

    def __init__(self, extra_patterns: Iterable[str] = ()) -> None:
        extra = [p.strip() for p in extra_patterns if p.strip() and not p.strip().startswith("#")]
        for pattern in extra:
            # A leading "!" would *un*-ignore a path in gitignore syntax, i.e. whitelist a secret.
            if pattern.startswith("!"):
                raise InvalidPatternError(
                    f"negated sensitive pattern {pattern!r} is not allowed; "
                    "config may only add sensitive patterns, never remove them"
                )
        self.patterns: tuple[str, ...] = DEFAULT_SENSITIVE_PATTERNS + tuple(extra)
        # Lower-case both sides: on Windows/macOS `.ENV` is the same file as `.env`.
        self._spec = pathspec.GitIgnoreSpec.from_lines(p.lower() for p in self.patterns)

    def is_sensitive(self, rel_path: str | PurePosixPath) -> bool:
        rel = str(rel_path).replace("\\", "/").lstrip("/")
        if not rel or rel == ".":
            return False
        return self._spec.match_file(rel.lower())


def resolve_in_workspace(root: Path, candidate: str | os.PathLike[str]) -> Path:
    """Return the canonical absolute path for `candidate`, or raise if it escapes `root`.

    `candidate` may be relative (to root) or absolute. Symlinks and `..` are resolved first, so
    the containment check is done on the real target, not on the spelling of the path.
    """
    root_real = root.resolve(strict=True)
    raw = os.fspath(candidate)
    if "\x00" in raw:
        raise PathOutsideWorkspaceError("path contains a NUL byte")
    path = Path(raw)
    if os.name == "nt" and ":" in (path.as_posix() if not path.is_absolute() else path.name):
        # Relative paths with a colon are either drive-relative (C:foo) or NTFS alternate data
        # streams (file.py:stream). Neither is something an edit tool should ever need.
        raise PathOutsideWorkspaceError(f"unsupported path form: {raw!r}")
    if not path.is_absolute():
        path = root_real / path
    real = path.resolve(strict=False)
    if real != root_real and not real.is_relative_to(root_real):
        raise PathOutsideWorkspaceError(f"{raw!r} resolves outside the workspace")
    return real


def to_workspace_relpath(root: Path, real_path: Path) -> str:
    """Workspace-relative POSIX path for an already-canonicalized absolute path."""
    return real_path.relative_to(root.resolve()).as_posix()
