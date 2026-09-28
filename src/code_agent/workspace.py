"""Workspace discovery: which directory is the repo root, and where the agent keeps its state."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

STATE_DIR_NAME = ".agent"


def run_git(
    root: Path, *args: str, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess:
    """Run a fixed, internally-constructed git command. Never used for model-generated commands."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        input=input_bytes,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _git_toplevel(start: Path) -> Path | None:
    try:
        proc = run_git(start, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return Path(proc.stdout.decode().strip()).resolve()


@dataclass(frozen=True)
class Workspace:
    root: Path
    is_git: bool

    @classmethod
    def discover(cls, start: Path) -> Workspace:
        start = start.resolve()
        top = _git_toplevel(start)
        if top is not None:
            return cls(root=top, is_git=True)
        return cls(root=start, is_git=False)

    @property
    def repo_id(self) -> str:
        # Stable per checkout location. Case-folded so C:\Repo and c:\repo agree on Windows.
        return hashlib.sha256(str(self.root).casefold().encode()).hexdigest()[:16]

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIR_NAME

    @property
    def index_path(self) -> Path:
        return self.state_dir / "index.db"

    def ensure_state_dir(self) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Self-ignoring directory: keeps agent state out of the user's `git status` without
        # touching their .gitignore.
        marker = self.state_dir / ".gitignore"
        if not marker.exists():
            marker.write_text("*\n", encoding="utf-8")
        return self.state_dir

    def git_revision(self) -> tuple[str | None, str | None]:
        """(branch, commit sha) or (None, None) outside git / on an unborn branch."""
        if not self.is_git:
            return None, None
        branch = run_git(self.root, "rev-parse", "--abbrev-ref", "HEAD")
        rev = run_git(self.root, "rev-parse", "HEAD")
        return (
            branch.stdout.decode().strip() if branch.returncode == 0 else None,
            rev.stdout.decode().strip() if rev.returncode == 0 else None,
        )
