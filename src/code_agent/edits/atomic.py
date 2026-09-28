"""Write a set of file changes all-or-nothing (as far as a filesystem allows).

A single file is replaced atomically with the classic recipe: write a temp file in the *same
directory* (so the rename cannot cross filesystems), fsync it, then `os.replace` it over the
target. Readers see either the old file or the new one, never a torn write.

Several files cannot be replaced in one atomic step, so we get as close as possible:

  phase 0  re-read every target and check it still has the expected hash (the user may have
           saved the file while the diff was on screen); abort before touching anything
  phase 1  write + fsync every temp file; a failure here (disk full, permissions) aborts with
           nothing replaced
  phase 2  rename temps into place; if one rename fails, already-replaced files are restored
           from the bytes captured in phase 0

A crash *during* phase 2 can still leave a subset applied. The change-set record is written
before phase 2 (see changesets.py), so `agent undo` can still restore those files.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from code_agent.hashing import sha256_bytes


@dataclass(frozen=True)
class FileWrite:
    path: Path
    expected_hash: str | None  # current content must hash to this; None = must not exist
    content: bytes | None  # new content; None = delete the file
    mode: int | None = None  # permission bits to keep (POSIX)


class WriteConflictError(RuntimeError):
    """A target changed since the plan was made. Nothing was written."""

    def __init__(self, paths: list[Path]) -> None:
        super().__init__(
            "changed on disk since the edit was planned: " + ", ".join(map(str, paths))
        )
        self.paths = paths


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":
        return  # Windows has no directory fsync; NTFS journals the rename metadata itself
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_temp(target: Path, content: bytes, mode: int | None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".agent-tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(name, mode)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(name)
        raise
    return Path(name)


def write_all(writes: Sequence[FileWrite]) -> None:
    # Phase 0: verify, and remember original bytes for rollback.
    originals: dict[Path, bytes | None] = {}
    conflicts: list[Path] = []
    for w in writes:
        current = w.path.read_bytes() if w.path.exists() else None
        originals[w.path] = current
        current_hash = None if current is None else sha256_bytes(current)
        if current_hash != w.expected_hash:
            conflicts.append(w.path)
    if conflicts:
        raise WriteConflictError(conflicts)

    # Phase 1: stage temp files.
    staged: list[tuple[FileWrite, Path | None]] = []
    try:
        for w in writes:
            temp = None if w.content is None else _write_temp(w.path, w.content, w.mode)
            staged.append((w, temp))
    except BaseException:
        for _, temp in staged:
            if temp is not None:
                with contextlib.suppress(OSError):
                    temp.unlink()
        raise

    # Phase 2: swap into place, rolling back on failure.
    done: list[FileWrite] = []
    try:
        for w, temp in staged:
            if temp is None:
                w.path.unlink()
            else:
                os.replace(temp, w.path)
            done.append(w)
    except BaseException:
        for w in reversed(done):
            with contextlib.suppress(OSError):
                original = originals[w.path]
                if original is None:
                    w.path.unlink()
                else:
                    restore = _write_temp(w.path, original, w.mode)
                    os.replace(restore, w.path)
        for _, temp in staged[len(done) :]:
            if temp is not None:
                with contextlib.suppress(OSError):
                    temp.unlink()
        raise
    for directory in {w.path.parent for w in writes}:
        _fsync_dir(directory)
