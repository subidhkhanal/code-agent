"""AST-aware chunking for Python, line windows for everything else.

Python rules (see docs/adr/0003-ast-chunking.md for the reasoning):

* Each top-level function is one chunk (decorators and directly-attached comments included).
* Each class produces a *header* chunk: decorators, the `class` line, docstring, class-level
  statements, and the signatures of its methods with bodies elided as `...`. Every method is then
  its own chunk with a qualified symbol (`Class.method`). Nested classes recurse.
* Top-level code that is not a definition (imports, constants, `if __name__ == ...`) is grouped
  into *contiguous runs*; each run is one `module` chunk. Contiguity matters: a chunk's content
  is always a verbatim slice of the file (except class headers), so the model can quote it back
  in a SEARCH block and it will match.
* Very long runs are split into overlapping line windows.

Newlines are normalized to `\\n` before parsing, and all line numbers are 1-based and inclusive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from code_agent.hashing import sha256_text

PYTHON = "python"


@dataclass(frozen=True)
class Chunk:
    kind: str  # module | function | class | method | text
    symbol: str | None  # qualified within the file: "Outer.Inner.method"
    name: str | None  # last component: "method"
    start_line: int
    end_line: int
    content: str

    @property
    def content_hash(self) -> str:
        return sha256_text(self.content)


@cache
def _parser() -> Parser:
    return Parser(Language(tree_sitter_python.language()))


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline does not start a new line
    return lines


def _end_row(node: Node) -> int:
    """Last row that contains the node's text (a node ending at column 0 ends on the row above)."""
    row = node.end_point.row
    if node.end_point.column == 0 and row > node.start_point.row:
        row -= 1
    return row


def _definition(node: Node) -> Node:
    """Unwrap `decorated_definition` to the function/class it decorates."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner is not None:
            return inner
    return node


def _name_of(defn: Node, source: bytes) -> str:
    # Slice our own copy of the source rather than using `Node.text`: the tree does not keep the
    # bytes passed to `parse()` alive, and reading them after they are freed crashes the process.
    name = defn.child_by_field_name("name")
    if name is None:
        return "<anonymous>"
    return source[name.start_byte : name.end_byte].decode("utf-8", "replace")


def _windows(
    lines: list[str], first_row: int, last_row: int, window: int, overlap: int,
    make: Callable[[int, int, str], Chunk],
) -> list[Chunk]:  # fmt: skip
    """Split rows [first_row, last_row] (0-based, inclusive) into overlapping windows."""
    out: list[Chunk] = []
    step = max(1, window - overlap)
    start = first_row
    while start <= last_row:
        end = min(last_row, start + window - 1)
        segment = lines[start : end + 1]
        if any(line.strip() for line in segment):
            out.append(make(start + 1, end + 1, "\n".join(segment)))
        if end == last_row:
            break
        start += step
    return out


def python_signature(outer: Node, defn: Node, lines: list[str]) -> str:
    """Source of a def/class header with the body elided: decorators + signature + `...`.

    Used for class header chunks now, and for signature-only context pruning later.
    """
    body = defn.child_by_field_name("body")
    start = outer.start_point.row
    if body is None or body.start_point.row <= defn.start_point.row:
        # One-liner (`def f(): return 1`): the whole thing is already as short as a signature.
        return "\n".join(lines[start : _end_row(outer) + 1])
    header = lines[start : body.start_point.row]
    def_line = lines[defn.start_point.row]
    indent = def_line[: len(def_line) - len(def_line.lstrip())]
    return "\n".join([*header, f"{indent}    ..."])


class _PythonChunker:
    def __init__(self, text: str, max_module_lines: int, window: int, overlap: int) -> None:
        self.lines = _split_lines(text)
        self.source = text.encode("utf-8")  # must outlive self.tree (see _name_of)
        self.tree = _parser().parse(self.source)
        self.max_module_lines = max_module_lines
        self.window = window
        self.overlap = overlap
        self.chunks: list[Chunk] = []

    def run(self) -> list[Chunk]:
        run: list[Node] = []  # current contiguous run of top-level non-definition nodes
        comments: list[Node] = []  # comments not yet assigned to a run or a definition

        for node in self.tree.root_node.named_children:
            if node.type == "comment":
                comments.append(node)
                continue
            defn = _definition(node)
            if defn.type in ("function_definition", "class_definition"):
                attached = self._attached_comments(comments, node)
                run.extend(comments[: len(comments) - len(attached)])
                comments.clear()
                self._flush_run(run)
                start_row = attached[0].start_point.row if attached else node.start_point.row
                if defn.type == "function_definition":
                    self._emit_function(node, defn, prefix="", kind="function", start_row=start_row)
                else:
                    self._emit_class(node, defn, prefix="", start_row=start_row)
            else:
                run.extend(comments)
                comments.clear()
                run.append(node)
        run.extend(comments)
        self._flush_run(run)
        return self.chunks

    @staticmethod
    def _attached_comments(comments: list[Node], definition: Node) -> list[Node]:
        """Trailing comments with no blank line between them and the definition."""
        attached: list[Node] = []
        next_row = definition.start_point.row
        for comment in reversed(comments):
            if _end_row(comment) + 1 != next_row:
                break
            attached.insert(0, comment)
            next_row = comment.start_point.row
        return attached

    def _flush_run(self, run: list[Node]) -> None:
        if not run:
            return
        first, last = run[0].start_point.row, max(_end_row(n) for n in run)
        run.clear()

        def make(start: int, end: int, content: str) -> Chunk:
            return Chunk("module", None, None, start, end, content)

        if last - first + 1 <= self.max_module_lines:
            self.chunks.append(make(first + 1, last + 1, self._slice(first, last)))
        else:
            self.chunks.extend(_windows(self.lines, first, last, self.window, self.overlap, make))

    def _slice(self, first_row: int, last_row: int) -> str:
        return "\n".join(self.lines[first_row : last_row + 1])

    def _emit_function(
        self, outer: Node, defn: Node, prefix: str, kind: str, start_row: int
    ) -> None:
        name = _name_of(defn, self.source)
        end = _end_row(outer)
        self.chunks.append(
            Chunk(kind, prefix + name, name, start_row + 1, end + 1, self._slice(start_row, end))
        )

    def _emit_class(self, outer: Node, defn: Node, prefix: str, start_row: int) -> None:
        name = _name_of(defn, self.source)
        qualified = prefix + name
        end = _end_row(outer)
        header_position = len(self.chunks)  # header goes before its methods in output order
        body = defn.child_by_field_name("body")

        if body is None or body.start_point.row <= defn.start_point.row:
            content = self._slice(start_row, end)  # one-line class
        else:
            parts = [self._slice(start_row, body.start_point.row - 1)]
            for member in body.named_children:
                member_defn = _definition(member)
                if member_defn.type == "function_definition":
                    parts.append(python_signature(member, member_defn, self.lines))
                    self._emit_function(
                        member, member_defn, prefix=qualified + ".", kind="method",
                        start_row=member.start_point.row,
                    )  # fmt: skip
                elif member_defn.type == "class_definition":
                    parts.append(python_signature(member, member_defn, self.lines))
                    self._emit_class(
                        member, member_defn, prefix=qualified + ".",
                        start_row=member.start_point.row,
                    )  # fmt: skip
                else:
                    parts.append(self._slice(member.start_point.row, _end_row(member)))
            content = "\n".join(parts)

        self.chunks.insert(
            header_position, Chunk("class", qualified, name, start_row + 1, end + 1, content)
        )


def chunk_python(
    text: str, *, max_module_lines: int = 150, window: int = 60, overlap: int = 10
) -> list[Chunk]:
    text = normalize_newlines(text)
    if not text.strip():
        return []
    return _PythonChunker(text, max_module_lines, window, overlap).run()


def chunk_text(text: str, *, window: int = 60, overlap: int = 10) -> list[Chunk]:
    lines = _split_lines(normalize_newlines(text))
    if not lines:
        return []

    def make(start: int, end: int, content: str) -> Chunk:
        return Chunk("text", None, None, start, end, content)

    return _windows(lines, 0, len(lines) - 1, window, overlap, make)


def chunk_file(
    text: str, language: str, *, max_module_lines: int = 150, window: int = 60, overlap: int = 10
) -> list[Chunk]:
    if language == PYTHON:
        return chunk_python(text, max_module_lines=max_module_lines, window=window, overlap=overlap)
    return chunk_text(text, window=window, overlap=overlap)
