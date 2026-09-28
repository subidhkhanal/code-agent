"""Fast Apply: turn parsed SEARCH/REPLACE blocks into a validated, in-memory change set.

Nothing here writes to disk; `atomic.write_changes` does that. Keeping planning pure means the
same plan can be applied to the shadow worktree first (M3) and to the real workspace only after
the user accepts, and a failed block never leaves a half-edited file behind.

Per file, in order:
1. Path policy: canonicalized path must be inside the workspace and not sensitive.
2. Base-hash check: the file's current hash must equal the hash recorded when the model last saw
   it. Mismatch -> STALE_FILE: re-retrieve and regenerate; never patch new content with an old
   plan. Editing a file the model never read -> NOT_READ.
3. Blocks for that file are applied in order to the in-memory text (later blocks see earlier
   edits). Each block: exact -> whitespace -> fuzzy match, unique or rejected.
4. Line terminators are preserved per line: untouched lines keep theirs, new lines get the
   file's dominant terminator; a missing final newline stays missing; a UTF-8 BOM is kept.
"""

from __future__ import annotations

import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from code_agent.edits.matching import (
    DEFAULT_FUZZY_THRESHOLD,
    Ambiguous,
    Match,
    MatchKind,
    NoMatch,
    find_block,
)
from code_agent.edits.parser import EditBlock
from code_agent.hashing import sha256_bytes
from code_agent.security.paths import (
    PathOutsideWorkspaceError,
    SensitivePathPolicy,
    resolve_in_workspace,
    to_workspace_relpath,
)

BOM = b"\xef\xbb\xbf"
_TERMINATOR = re.compile(r"\r\n|\n|\r")
MAX_FEEDBACK_LINES = 12


class ApplyErrorCode(StrEnum):
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    STALE_FILE = "STALE_FILE"
    NOT_READ = "NOT_READ"
    PATH_DENIED = "PATH_DENIED"
    FILE_EXISTS = "FILE_EXISTS"
    FILE_MISSING = "FILE_MISSING"
    ENCODING = "ENCODING"
    EMPTY_SEARCH = "EMPTY_SEARCH"


@dataclass(frozen=True)
class BlockResult:
    block: EditBlock
    error: ApplyErrorCode | None = None
    kind: MatchKind | str | None = None  # exact | whitespace | fuzzy | create
    similarity: float | None = None
    start_line: int | None = None  # 1-based, in the file as it was before this block
    end_line: int | None = None
    message: str = ""  # human/model-readable explanation, used as retry feedback

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class FileChange:
    rel_path: str
    abs_path: Path
    before: bytes | None  # None: the change creates the file
    after: bytes
    mode: int | None

    @property
    def before_hash(self) -> str | None:
        return None if self.before is None else sha256_bytes(self.before)

    @property
    def after_hash(self) -> str:
        return sha256_bytes(self.after)


@dataclass
class ApplyPlan:
    results: list[BlockResult] = field(default_factory=list)
    changes: dict[str, FileChange] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def errors(self) -> list[BlockResult]:
        return [r for r in self.results if not r.ok]

    def feedback(self) -> str:
        """Error report to send back to the model when the plan is rejected."""
        return "\n\n".join(f"Edit block #{r.block.index + 1} ({r.block.path}): {r.message}"
                           for r in self.errors)  # fmt: skip


# -- text model ---------------------------------------------------------------------------------


@dataclass
class _Document:
    lines: list[str]  # content without terminators
    ends: list[str]  # terminator per line ("" for a final line without newline)
    bom: bool

    @classmethod
    def decode(cls, data: bytes) -> _Document:
        bom = data.startswith(BOM)
        text = data[len(BOM) :].decode("utf-8") if bom else data.decode("utf-8")
        lines: list[str] = []
        ends: list[str] = []
        pos = 0
        for m in _TERMINATOR.finditer(text):
            lines.append(text[pos : m.start()])
            ends.append(m.group())
            pos = m.end()
        if pos < len(text):
            lines.append(text[pos:])
            ends.append("")
        return cls(lines, ends, bom)

    @property
    def dominant_ending(self) -> str:
        crlf = sum(1 for e in self.ends if e == "\r\n")
        lf = sum(1 for e in self.ends if e == "\n")
        return "\r\n" if crlf > lf else "\n"

    def replace(self, start: int, end: int, new_lines: list[str]) -> None:
        """Replace lines [start, end). New lines get the dominant terminator, except the last
        one, which inherits the terminator of the last replaced line (so a file without a final
        newline keeps not having one, and a lone CRLF line in an LF file stays CRLF)."""
        at_eof = end == len(self.lines)
        old_last_end = self.ends[end - 1] if end > start else None
        new_ends = [self.dominant_ending] * len(new_lines)
        if new_lines and old_last_end is not None:
            new_ends[-1] = old_last_end
        self.lines[start:end] = new_lines
        self.ends[start:end] = new_ends
        if not new_lines and at_eof and old_last_end == "" and start > 0:
            self.ends[start - 1] = ""  # deleted the unterminated last line(s)

    def encode(self) -> bytes:
        body = "".join(line + end for line, end in zip(self.lines, self.ends, strict=True))
        return (BOM if self.bom else b"") + body.encode("utf-8")


def _block_lines(text: str) -> list[str]:
    """SEARCH/REPLACE text -> lines without terminators."""
    lines = _TERMINATOR.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _excerpt(lines: list[str], start: int, count: int) -> str:
    shown = lines[start : start + min(count, MAX_FEEDBACK_LINES)]
    body = "\n".join(f"{start + i + 1:>5} | {line}" for i, line in enumerate(shown))
    return body + ("\n  ..." if count > MAX_FEEDBACK_LINES else "")


# -- planning -----------------------------------------------------------------------------------


class Planner:
    def __init__(
        self,
        root: Path,
        sensitive: SensitivePathPolicy | None = None,
        *,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> None:
        self.root = root.resolve()
        self.sensitive = sensitive or SensitivePathPolicy()
        self.fuzzy_threshold = fuzzy_threshold

    def plan(self, blocks: Iterable[EditBlock], base_hashes: Mapping[str, str]) -> ApplyPlan:
        """`base_hashes`: workspace-relative path -> sha256 of the bytes the model was shown."""
        plan = ApplyPlan()
        by_file: dict[str, list[EditBlock]] = {}
        denied: list[BlockResult] = []
        resolved: dict[str, Path] = {}
        for block in blocks:
            try:
                real = resolve_in_workspace(self.root, block.path)
                rel = to_workspace_relpath(self.root, real)
            except PathOutsideWorkspaceError as exc:
                denied.append(BlockResult(block, ApplyErrorCode.PATH_DENIED, message=str(exc)))
                continue
            if self.sensitive.is_sensitive(rel):
                denied.append(
                    BlockResult(block, ApplyErrorCode.PATH_DENIED,
                                message=f"{rel} is a protected path and cannot be edited")
                )  # fmt: skip
                continue
            resolved[rel] = real
            by_file.setdefault(rel, []).append(block)

        plan.results.extend(denied)
        for rel, file_blocks in by_file.items():
            self._plan_file(plan, rel, resolved[rel], file_blocks, base_hashes.get(rel))
        plan.results.sort(key=lambda r: r.block.index)
        if not plan.ok:
            plan.changes.clear()  # all-or-nothing: never hand out a partial change set
        return plan

    def _plan_file(
        self,
        plan: ApplyPlan,
        rel: str,
        real: Path,
        blocks: list[EditBlock],
        base_hash: str | None,
    ) -> None:
        def reject_all(code: ApplyErrorCode, message: str) -> None:
            plan.results.extend(BlockResult(b, code, message=message) for b in blocks)

        if not real.exists():
            self._plan_new_file(plan, rel, real, blocks)
            return
        before = real.read_bytes()
        if base_hash is None:
            reject_all(ApplyErrorCode.NOT_READ,
                       f"{rel} was not read in this task; read it before editing it")  # fmt: skip
            return
        if sha256_bytes(before) != base_hash:
            reject_all(ApplyErrorCode.STALE_FILE,
                       f"{rel} changed on disk after you read it; re-read it and regenerate "
                       "the edit against the current content")  # fmt: skip
            return
        try:
            doc = _Document.decode(before)
        except UnicodeDecodeError:
            reject_all(ApplyErrorCode.ENCODING, f"{rel} is not valid UTF-8")
            return

        for block in blocks:
            plan.results.append(self._apply_block(doc, block))
        after = doc.encode()
        if after != before:
            mode = stat.S_IMODE(real.stat().st_mode)
            plan.changes[rel] = FileChange(rel, real, before, after, mode)

    def _plan_new_file(
        self, plan: ApplyPlan, rel: str, real: Path, blocks: list[EditBlock]
    ) -> None:
        first, rest = blocks[0], blocks[1:]
        if first.search.strip():
            plan.results.extend(
                BlockResult(b, ApplyErrorCode.FILE_MISSING,
                            message=f"{rel} does not exist; to create it use an empty SEARCH")
                for b in blocks
            )  # fmt: skip
            return
        doc = _Document([], [], bom=False)
        doc.replace(0, 0, _block_lines(first.replace))
        plan.results.append(BlockResult(first, kind="create", start_line=1,
                                        end_line=len(doc.lines), message="created"))  # fmt: skip
        for block in rest:
            plan.results.append(self._apply_block(doc, block))
        plan.changes[rel] = FileChange(rel, real, None, doc.encode(), None)

    def _apply_block(self, doc: _Document, block: EditBlock) -> BlockResult:
        needle = _block_lines(block.search)
        if not any(line.strip() for line in needle):
            return BlockResult(block, ApplyErrorCode.EMPTY_SEARCH,
                               message="SEARCH is empty but the file already exists; quote the "
                               "exact lines to replace")  # fmt: skip
        outcome = find_block(doc.lines, needle, fuzzy_threshold=self.fuzzy_threshold)
        if isinstance(outcome, Ambiguous):
            where = ", ".join(str(s + 1) for s in outcome.starts[:5])
            return BlockResult(block, ApplyErrorCode.AMBIGUOUS_MATCH,
                               message=f"SEARCH matches {len(outcome.starts)} places "
                               f"({outcome.kind} match at lines {where}); include more "
                               "surrounding lines so it matches exactly one place")  # fmt: skip
        if isinstance(outcome, NoMatch):
            message = "SEARCH does not match the file"
            if outcome.closest_start is not None and outcome.closest_similarity > 0.5:
                message += (
                    f". Closest lines (similarity {outcome.closest_similarity:.2f}):\n"
                    + _excerpt(doc.lines, outcome.closest_start, len(needle))
                )
            return BlockResult(block, ApplyErrorCode.NO_MATCH, message=message)

        assert isinstance(outcome, Match)
        replacement = outcome.reindent(_block_lines(block.replace))
        doc.replace(outcome.start, outcome.end, replacement)
        return BlockResult(
            block, kind=outcome.kind, similarity=outcome.similarity,
            start_line=outcome.start + 1, end_line=outcome.end, message=f"{outcome.kind} match",
        )  # fmt: skip
