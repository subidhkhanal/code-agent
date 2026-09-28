"""Which files get indexed.

Layers, applied in order (a file must pass all of them):

1. VCS ignore rules. In a git repo we ask git itself (`ls-files` / `check-ignore`), which gets
   nested `.gitignore`s, `.git/info/exclude` and global excludes exactly right. Outside git we
   fall back to the root `.gitignore` via pathspec.
2. Built-in excludes (virtualenvs, caches, build output, lock files) + `.agentignore` + config.
3. The sensitive-path policy (security/paths.py). Sensitive files are dropped even if tracked.
4. File type: `.py`/`.pyi`, or a configured text extension/filename.
5. Content: symlinks, files over the size limit, binary files and non-UTF-8 files are skipped.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pathspec

from code_agent.config import IndexConfig
from code_agent.index.chunker import PYTHON
from code_agent.security.paths import SensitivePathPolicy
from code_agent.workspace import STATE_DIR_NAME, Workspace, run_git

log = logging.getLogger(__name__)

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git/", f"{STATE_DIR_NAME}/",
    ".venv/", "venv/", ".env/", "env/", "node_modules/", "site-packages/", "__pypackages__/",
    "__pycache__/", "*.py[cod]", ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".tox/",
    ".nox/", ".hypothesis/", ".ipynb_checkpoints/",
    "build/", "dist/", "*.egg-info/", ".eggs/", "htmlcov/", ".coverage",
    "*.min.js", "*.min.css", "*.map",
    "*.lock", "package-lock.json", "pnpm-lock.yaml",
)  # fmt: skip

AGENTIGNORE = ".agentignore"
BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class SourceFile:
    rel_path: str  # workspace-relative, POSIX separators
    abs_path: Path
    size: int
    mtime_ns: int
    language: str


def detect_language(rel_path: str, cfg: IndexConfig) -> str | None:
    name = rel_path.rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()
    if suffix in (".py", ".pyi"):
        return PYTHON
    if suffix in cfg.text_extensions:
        return suffix.lstrip(".")
    if name in cfg.text_filenames:
        return "text"
    return None


def read_source_text(data: bytes) -> str | None:
    """Decode file bytes for indexing, or None for binary / non-UTF-8 content."""
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


class FileSelector:
    def __init__(self, workspace: Workspace, cfg: IndexConfig) -> None:
        self.workspace = workspace
        self.cfg = cfg
        self.sensitive = SensitivePathPolicy(cfg.extra_sensitive)
        self._excludes = pathspec.GitIgnoreSpec.from_lines(
            [*DEFAULT_EXCLUDES, *self._read_agentignore(), *cfg.extra_ignore]
        )
        self._fallback_gitignore: pathspec.GitIgnoreSpec | None = None
        if not workspace.is_git:
            gitignore = workspace.root / ".gitignore"
            lines = (
                gitignore.read_text("utf-8", "replace").splitlines() if gitignore.is_file() else []
            )
            self._fallback_gitignore = pathspec.GitIgnoreSpec.from_lines(lines)

    def _read_agentignore(self) -> list[str]:
        path = self.workspace.root / AGENTIGNORE
        if not path.is_file():
            return []
        # Negations are dropped: .agentignore may only narrow what gets indexed.
        lines = path.read_text("utf-8", "replace").splitlines()
        return [ln for ln in lines if not ln.strip().startswith("!")]

    # -- policy -------------------------------------------------------------------------------

    def is_candidate(self, rel_path: str) -> bool:
        """Layers 2-4: cheap, path-only checks."""
        if self._excludes.match_file(rel_path) or self.sensitive.is_sensitive(rel_path):
            return False
        return detect_language(rel_path, self.cfg) is not None

    def vcs_ignored(self, rel_paths: Iterable[str]) -> set[str]:
        """Layer 1 for an explicit list of paths (used by the watcher)."""
        paths = list(rel_paths)
        if not paths:
            return set()
        if self._fallback_gitignore is not None:
            return {p for p in paths if self._fallback_gitignore.match_file(p)}
        proc = run_git(
            self.workspace.root, "check-ignore", "-z", "--stdin",
            input_bytes=b"\x00".join(p.encode() for p in paths) + b"\x00",
        )  # fmt: skip
        # Exit code 1 means "none ignored"; anything above that is a real error.
        if proc.returncode > 1:
            log.warning("git check-ignore failed: %s", proc.stderr.decode(errors="replace"))
            return set()
        return {p for p in proc.stdout.decode().split("\x00") if p}

    # -- discovery ----------------------------------------------------------------------------

    def _vcs_listing(self) -> list[str]:
        if self._fallback_gitignore is not None:
            return self._walk_fallback()
        proc = run_git(
            self.workspace.root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git ls-files failed: {proc.stderr.decode(errors='replace')}")
        return sorted({p for p in proc.stdout.decode().split("\x00") if p})

    def _walk_fallback(self) -> list[str]:
        assert self._fallback_gitignore is not None
        root = self.workspace.root
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            rel_dir = "" if rel_dir == "." else rel_dir + "/"
            # Prune ignored directories early so we never descend into .venv etc.
            dirnames[:] = [
                d for d in dirnames
                if not self._excludes.match_file(f"{rel_dir}{d}/")
                and not self._fallback_gitignore.match_file(f"{rel_dir}{d}/")
            ]  # fmt: skip
            for name in filenames:
                rel = f"{rel_dir}{name}"
                if not self._fallback_gitignore.match_file(rel):
                    out.append(rel)
        return sorted(out)

    def stat(self, rel_path: str) -> SourceFile | None:
        """Layer 5 (metadata part): returns None for anything that must not be indexed."""
        language = detect_language(rel_path, self.cfg)
        if language is None:
            return None
        abs_path = self.workspace.root / rel_path
        try:
            # Symlinks are skipped outright: they can point outside the workspace or at a
            # sensitive file under an innocent name, and in-repo targets are indexed directly.
            if abs_path.is_symlink():
                return None
            st = abs_path.stat()
        except OSError:
            return None
        if not abs_path.is_file() or st.st_size > self.cfg.max_file_bytes:
            return None
        return SourceFile(rel_path, abs_path, st.st_size, st.st_mtime_ns, language)

    def scan(self) -> list[SourceFile]:
        files: list[SourceFile] = []
        for rel in self._vcs_listing():
            if not self.is_candidate(rel):
                continue
            source = self.stat(rel)
            if source is not None:
                files.append(source)
        return files
