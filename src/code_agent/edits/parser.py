"""Streaming parser for SEARCH/REPLACE edit blocks.

The model writes edits in this format (see docs/adr/0004-edit-format.md):

    auth/tokens.py
    <<<<<<< SEARCH
        return token.expires_at > 0
    =======
        return token.expires_at > time.time()
    >>>>>>> REPLACE

Tokens arrive in arbitrary fragments, so the parser buffers until it has a complete line and runs
a small state machine over lines. A block is emitted the moment its `>>>>>>> REPLACE` line
arrives, so the apply engine can start on block 1 while the model is still writing block 2.

Everything outside blocks is prose and is passed through as `Text` events for display.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto

SEARCH_MARK = re.compile(r"^\s*<{5,9} ?SEARCH\s*$")
DIVIDER_MARK = re.compile(r"^\s*={5,9}\s*$")
REPLACE_MARK = re.compile(r"^\s*>{5,9} ?REPLACE\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class EditBlock:
    path: str
    search: str  # "\n"-joined lines, each line ending with "\n" (empty string for new files)
    replace: str
    index: int  # 0-based position in the response, for error messages


@dataclass(frozen=True)
class Text:
    """Prose outside any block (one complete line)."""

    text: str


@dataclass(frozen=True)
class BlockStarted:
    path: str
    index: int


@dataclass(frozen=True)
class ParseError:
    message: str
    index: int


ParseEvent = Text | BlockStarted | EditBlock | ParseError


class _State(Enum):
    OUTSIDE = auto()
    SEARCH = auto()
    REPLACE = auto()


def _clean_path(line: str) -> str | None:
    """Interpret a line as a file path, tolerating common decoration (`**a.py**`, `# a.py`)."""
    candidate = line.strip().strip("`*").strip()
    candidate = re.sub(r"^(#+\s*|file:\s*|path:\s*)", "", candidate, flags=re.IGNORECASE).strip()
    candidate = candidate.rstrip(":").strip("`*").strip()
    if not candidate or " " in candidate or FENCE.match(candidate):
        return None
    return candidate


class StreamingEditParser:
    """Feed model output with `feed()`, then call `finish()`. Both yield `ParseEvent`s."""

    def __init__(self) -> None:
        self._buffer = ""
        self._state = _State.OUTSIDE
        self._last_path_candidate: str | None = None  # most recent path-looking prose line
        self._previous_path: str | None = None  # path of the last block (models omit repeats)
        self._path: str = ""
        self._search: list[str] = []
        self._replace: list[str] = []
        self._count = 0

    def feed(self, fragment: str) -> Iterator[ParseEvent]:
        self._buffer += fragment.replace("\r\n", "\n")
        while (newline := self._buffer.find("\n")) != -1:
            line, self._buffer = self._buffer[: newline + 1], self._buffer[newline + 1 :]
            yield from self._line(line)

    def finish(self) -> Iterator[ParseEvent]:
        if self._buffer:
            line, self._buffer = self._buffer, ""
            yield from self._line(line if line.endswith("\n") else line + "\n")
        if self._state is not _State.OUTSIDE:
            where = "=======" if self._state is _State.SEARCH else ">>>>>>> REPLACE"
            yield ParseError(f"block for {self._path!r} ended without {where}", self._count)
            self._state = _State.OUTSIDE

    def _line(self, line: str) -> Iterator[ParseEvent]:
        bare = line.rstrip("\n")
        if self._state is _State.OUTSIDE:
            if SEARCH_MARK.match(bare):
                path = self._last_path_candidate or self._previous_path
                if path is None:
                    yield ParseError("SEARCH block has no file path before it", self._count)
                    path = ""
                self._path, self._search, self._replace = path, [], []
                self._state = _State.SEARCH
                yield BlockStarted(path, self._count)
                return
            if bare.strip() and not FENCE.match(bare):
                self._last_path_candidate = _clean_path(bare)
            yield Text(line)
        elif self._state is _State.SEARCH:
            if DIVIDER_MARK.match(bare):
                self._state = _State.REPLACE
            elif SEARCH_MARK.match(bare) or REPLACE_MARK.match(bare):
                yield ParseError(f"unexpected marker {bare.strip()!r} in SEARCH", self._count)
            else:
                self._search.append(line)
        else:  # REPLACE
            if REPLACE_MARK.match(bare):
                block = EditBlock(
                    self._path, "".join(self._search), "".join(self._replace), self._count
                )
                self._count += 1
                self._previous_path = self._path or self._previous_path
                self._last_path_candidate = None
                self._state = _State.OUTSIDE
                if block.path:
                    yield block
            elif DIVIDER_MARK.match(bare) or SEARCH_MARK.match(bare):
                yield ParseError(f"unexpected marker {bare.strip()!r} in REPLACE", self._count)
            else:
                self._replace.append(line)


def parse_edit_blocks(chunks: Iterable[str]) -> Iterator[ParseEvent]:
    """Convenience wrapper: parse an iterable of text fragments (e.g. an LLM stream)."""
    parser = StreamingEditParser()
    for chunk in chunks:
        yield from parser.feed(chunk)
    yield from parser.finish()


def parse_all(text: str) -> tuple[list[EditBlock], list[ParseError]]:
    blocks: list[EditBlock] = []
    errors: list[ParseError] = []
    for event in parse_edit_blocks([text]):
        if isinstance(event, EditBlock):
            blocks.append(event)
        elif isinstance(event, ParseError):
            errors.append(event)
    return blocks, errors
