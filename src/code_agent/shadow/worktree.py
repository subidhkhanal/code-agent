"""A hidden git worktree that mirrors the user's working tree (see ADR 0006).

`git worktree add --detach` gives a second checkout of HEAD that shares the object store, so
creating it is cheap and it never touches the user's branch, index or working files. On top of
HEAD we copy the user's *uncommitted* state: modified and untracked-but-not-ignored files are
copied in, deleted files are removed. Edits are then written into this copy for validation;
the real workspace is untouched until the user accepts.

The worktree is reused across tasks in a session (`reset()` brings it back to the user's current
state) and removed on `close()`. A worktree left behind by a crash is pruned the next time.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from code_agent.workspace import Workspace, run_git

SHADOW_PREFIX = "code-agent-shadow-"


class ShadowUnavailableError(RuntimeError):
    pass


def _rmtree(path: Path) -> None:
    def make_writable(func, target, _exc) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if path.exists():
        shutil.rmtree(path, onexc=make_writable)


def _git(root: Path, *args: str) -> str:
    proc = run_git(root, *args)
    if proc.returncode != 0:
        raise ShadowUnavailableError(
            f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode(errors="replace")


class ShadowWorktree:
    def __init__(self, workspace: Workspace) -> None:
        if not workspace.is_git:
            raise ShadowUnavailableError("the workspace is not a git repository")
        self.workspace = workspace
        self.path: Path | None = None

    @property
    def root(self) -> Path:
        if self.path is None:
            raise ShadowUnavailableError("shadow worktree not created")
        return self.path

    def _head(self) -> str:
        return _git(self.workspace.root, "rev-parse", "HEAD").strip()

    def create(self) -> Path:
        if self.path is not None:
            return self.path
        run_git(self.workspace.root, "worktree", "prune")  # forget worktrees whose dirs vanished
        path = Path(tempfile.mkdtemp(prefix=SHADOW_PREFIX)) / "tree"
        _git(self.workspace.root, "worktree", "add", "--detach", "--quiet", str(path), self._head())
        self.path = path
        self.sync_uncommitted()
        return path

    def reset(self) -> None:
        """Back to the user's current state: their HEAD plus their uncommitted changes."""
        root = self.root
        _git(root, "checkout", "--detach", "--quiet", "--force", self._head())
        _git(root, "reset", "--hard", "--quiet")
        _git(root, "clean", "-fdq")  # untracked files; ignored ones (caches) are kept
        self.sync_uncommitted()

    def sync_uncommitted(self) -> list[str]:
        """Copy modified and untracked (non-ignored) files in, remove deleted ones."""
        user, shadow = self.workspace.root, self.root
        changed = _git(user, "diff", "--name-only", "-z", "HEAD").split("\x00")
        untracked = _git(user, "ls-files", "--others", "--exclude-standard", "-z").split("\x00")
        synced: list[str] = []
        for rel in {p for p in changed + untracked if p}:
            source, target = user / rel, shadow / rel
            if source.is_symlink():
                continue
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                synced.append(rel)
            elif not source.exists() and target.exists():
                target.unlink()
                synced.append(rel)
        return sorted(synced)

    def write(self, rel_path: str, content: bytes) -> None:
        target = self.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def close(self) -> None:
        if self.path is None:
            return
        run_git(self.workspace.root, "worktree", "remove", "--force", str(self.path))
        _rmtree(self.path.parent)
        run_git(self.workspace.root, "worktree", "prune")
        self.path = None

    def __enter__(self) -> ShadowWorktree:
        self.create()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
